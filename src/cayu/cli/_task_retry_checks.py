"""Advisory source checks for the boundary between task and event retries."""

from __future__ import annotations

import ast
from collections.abc import Iterable
from pathlib import Path

from cayu.runtime.checks import DiagnosticSeverity, ProjectDiagnostic


def task_retry_policy_diagnostics(
    root: Path, paths: Iterable[Path]
) -> tuple[ProjectDiagnostic, ...]:
    """Match explicit bounded workers to unkeyed task creation of the same type.

    This is intentionally a source hint, not a proof of an application's retry
    behavior. Dynamic declarations and task identity are never guessed.
    """
    bounded_types: set[str] = set()
    creations: list[tuple[str, str]] = []
    for path in paths:
        if path.is_symlink():
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, SyntaxError):
            continue
        imports: dict[str, str] = {}
        constants: dict[str, ast.expr] = {}
        for statement in tree.body:
            if isinstance(statement, ast.ImportFrom) and statement.level == 0:
                for item in statement.names:
                    imports[item.asname or item.name] = f"{statement.module}.{item.name}"
            elif isinstance(statement, ast.Import):
                for item in statement.names:
                    imports[item.asname or item.name.split(".")[0]] = (
                        item.name if item.asname else item.name.split(".")[0]
                    )
            elif isinstance(statement, ast.Assign):
                for target in statement.targets:
                    if isinstance(target, ast.Name):
                        constants[target.id] = statement.value

        # Do not attribute a locally rebound or shadowed name to Cayu. This
        # deliberately favors missing a dynamic call over a false diagnostic.
        rebound: set[str] = set()
        bindings: dict[str, int] = {}
        for node in ast.walk(tree):
            name = None
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                name = node.id
            elif isinstance(node, ast.arg):
                name = node.arg
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = node.name
            if name is not None:
                rebound.add(name)
                bindings[name] = bindings.get(name, 0) + 1
        constants = {name: value for name, value in constants.items() if bindings.get(name) == 1}
        imports = {name: target for name, target in imports.items() if name not in rebound}

        for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
            name = _qualified(call.func, imports)
            keywords = {item.arg: item.value for item in call.keywords}
            if None in keywords:  # **kwargs may carry the policy or query.
                continue
            if name == "cayu.tasks.run_task_worker":
                timeout = _literal(keywords.get("execution_timeout_s"), constants)
                query = keywords.get("query")
                if (
                    not isinstance(timeout, (int, float))
                    or isinstance(timeout, bool)
                    or timeout <= 0
                ):
                    continue
                if (
                    not isinstance(query, ast.Call)
                    or _qualified(query.func, imports) != "cayu.tasks.TaskQuery"
                ):
                    continue
                kind = next(
                    (
                        _literal(item.value, constants)
                        for item in query.keywords
                        if item.arg == "type"
                    ),
                    None,
                )
                if isinstance(kind, str):
                    bounded_types.add(kind)
            elif name == "cayu.tasks.TaskCreate":
                if any(
                    (value := keywords.get(key)) is not None
                    and not (isinstance(value, ast.Constant) and value.value is None)
                    for key in ("retry_key", "retry_policy")
                ):
                    continue
                kind = _literal(keywords.get("type"), constants)
                if isinstance(kind, str):
                    creations.append((kind, f"{path.relative_to(root).as_posix()}:{call.lineno}"))
    return tuple(
        ProjectDiagnostic(
            code="TASK_EVENT_RETRY_POLICY_UNDECLARED",
            severity=DiagnosticSeverity.WARNING,
            subject="task",
            path=path,
            message=(
                f"Task type {kind!r} has a bounded worker but this creation has no event "
                "retry policy. The handler timeout ends only this task."
            ),
            hint=(
                "If the application resubmits failed events, choose an event retry policy: "
                "submit once with TaskRetryPolicy, or use a stable event retry_key and an "
                "explicit TaskRetryBudget. Choose limits from the workload's contract; "
                "TaskRetryBudget.unlimited() declares intentional unlimited resubmission."
            ),
            parameters={"task_type": kind},
            tags=("authoring", "configuration"),
            documentation_anchor="cayu guide long-running#event-retry-policy",
            verification_command="cayu check --json",
        )
        for kind, path in creations
        if kind in bounded_types
    )


def _qualified(node: ast.expr, imports: dict[str, str]) -> str:
    if isinstance(node, ast.Name):
        return imports.get(node.id, "")
    if isinstance(node, ast.Attribute):
        return f"{_qualified(node.value, imports)}.{node.attr}"
    return ""


def _literal(node: ast.expr | None, constants: dict[str, ast.expr]) -> object:
    if isinstance(node, ast.Name):
        node = constants.get(node.id)
    return node.value if isinstance(node, ast.Constant) else None
