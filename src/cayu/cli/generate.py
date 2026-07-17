from __future__ import annotations

import argparse
import ast
import hashlib
import json
import keyword
import re
import shutil
import sys
import tempfile
import textwrap
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from cayu.cli.project import ProjectError, resolve_project
from cayu.cli.scaffold import (
    GENERATED_IMPORTS_END,
    GENERATED_IMPORTS_START,
    GENERATED_REGISTRATIONS_END,
    GENERATED_REGISTRATIONS_START,
)
from cayu.core.agents import AgentAuthoringState

_IDENTIFIER_RE = re.compile(r"[a-z][a-z0-9_]*")
GENERATOR_PLAN_SCHEMA_VERSION = "3"


class GeneratorEdit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    operation: Literal["create", "update_region"]
    content: str
    content_sha256: str
    preimage_sha256: str | None = None
    anchor: str | None = None


class GeneratorPrecondition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    content_sha256: str


class GeneratorPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["3"] = GENERATOR_PLAN_SCHEMA_VERSION
    status: Literal["ready", "conflict", "manual_action_required", "already_present"]
    slice_name: str
    tool_name: str
    effect: str
    authoring_state: AgentAuthoringState = AgentAuthoringState.UNFINISHED_GENERATED_TRACER_BULLET
    edits: tuple[GeneratorEdit, ...]
    preconditions: tuple[GeneratorPrecondition, ...] = ()
    conflicts: tuple[dict[str, str], ...] = ()
    verification_commands: tuple[str, ...]


class GeneratorApplyError(RuntimeError):
    """The planned generator transaction could not be applied safely."""


class _GeneratedPathError(ValueError):
    pass


@dataclass(frozen=True)
class _AgentRegistrationInspection:
    origins_by_name: dict[str, tuple[tuple[str, str], ...]]
    source_preconditions: tuple[GeneratorPrecondition, ...]
    unresolved_origins: frozenset[tuple[str, str]]


@dataclass(frozen=True)
class _AgentModuleSnapshot:
    tree: ast.Module
    precondition: GeneratorPrecondition


@dataclass(frozen=True)
class _RegionStatement:
    key: str
    source: str


def add_generate_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "generate",
        help="Plan or add reviewable generated Cayu application slices.",
    )
    generators = parser.add_subparsers(dest="generate_command", required=True)
    slice_parser = generators.add_parser(
        "slice",
        help="Add one agent, typed tool, runtime test, and trajectory eval.",
    )
    slice_parser.add_argument("name", help="snake_case agent/slice name.")
    slice_parser.add_argument("--tool", required=True, help="snake_case tool name.")
    slice_parser.add_argument(
        "--effect",
        choices=("none", "idempotent", "external"),
        required=True,
        help="Declared ToolEffect. See `cayu guide tool-effects` for the decision table.",
    )
    slice_parser.add_argument("--dry-run", action="store_true", help="Plan without writes.")
    slice_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the versioned plan without writing files.",
    )


def run_generate(args: argparse.Namespace) -> int:
    if args.generate_command != "slice":
        return 2
    try:
        plan = plan_slice(
            name=args.name,
            tool_name=args.tool,
            effect=args.effect,
        )
    except (ProjectError, ValueError, OSError) as exc:
        if args.json:
            print(
                json.dumps(
                    {
                        "schema_version": GENERATOR_PLAN_SCHEMA_VERSION,
                        "error": {"code": "GENERATOR_PLAN_FAILED", "message": str(exc)},
                    },
                    sort_keys=True,
                )
            )
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 2

    should_apply = not args.dry_run and not args.json and plan.status == "ready"
    if should_apply:
        try:
            apply_slice_plan(plan)
        except (GeneratorApplyError, ProjectError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    if args.json:
        print(plan.model_dump_json(indent=2))
    else:
        print(_render_plan(plan, applied=should_apply))
    return 0 if plan.status in {"ready", "already_present"} else 1


def plan_slice(*, name: str, tool_name: str, effect: str) -> GeneratorPlan:
    name = _identifier(name, "slice name")
    tool_name = _identifier(tool_name, "tool name")
    tool_name_constant = f"{_constant_name(tool_name)}_TOOL_NAME"
    if effect not in {"none", "idempotent", "external"}:
        raise ValueError("effect must be none, idempotent, or external.")
    project = resolve_project(command="cayu generate")
    root = project.root
    app_path = _generated_path(root, "app.py")
    if not app_path.is_file():
        raise ValueError("Generated registration target is missing: app.py.")
    app_content = app_path.read_bytes()
    app_source = app_content.decode("utf-8")
    app_precondition = GeneratorPrecondition(
        path="app.py",
        content_sha256=_sha256(app_content),
    )
    verification = (
        "cayu inspect --json",
        "cayu check --json",
        f"pytest tests/test_{name}.py",
        f"cayu eval run evals.{name}:build_eval",
    )
    independent = _slice_files(name=name, tool_name=tool_name, effect=effect)
    tool_imports = [_class_name(tool_name) + "Tool"]
    if effect == "external":
        tool_imports.append(tool_name_constant)
    import_lines = [
        f"from agents.{name} import {_constant_name(name)}_AGENT",
        f"from tools.{tool_name} import {', '.join(tool_imports)}",
    ]
    if effect == "external":
        import_lines.append("from cayu import AlwaysRequireApprovalToolPolicy")
    agent_constant = f"{_constant_name(name)}_AGENT"
    tool_instance = f"{_class_name(tool_name)}Tool()"
    if effect == "external":
        registration = (
            "app.register_agent(\n"
            f"    {agent_constant},\n"
            f"    tools=[{tool_instance}],\n"
            "    tool_policy=AlwaysRequireApprovalToolPolicy("
            f"tools=[{tool_name_constant}]),\n"
            ")"
        )
    else:
        registration = f"app.register_agent({agent_constant}, tools=[{tool_instance}])"

    conflicts: list[dict[str, str]] = []
    edits: list[GeneratorEdit] = []
    preconditions: dict[str, GeneratorPrecondition] = {}
    proposed_origin = (f"agents.{name}", f"{_constant_name(name)}_AGENT")
    agent_inspection = _inspect_registered_agents(root, app_source)
    registered_origins = list(agent_inspection.origins_by_name.get(name, ()))
    if _region_contains_statement(
        app_source,
        start=GENERATED_REGISTRATIONS_START,
        end=GENERATED_REGISTRATIONS_END,
        statement=registration,
    ):
        with suppress(ValueError):
            registered_origins.remove(proposed_origin)
    conflicting_origins = set(registered_origins)
    if conflicting_origins:
        rendered_origins = ", ".join(
            _render_agent_origin(origin) for origin in sorted(conflicting_origins)
        )
        conflicts.append(
            {
                "path": "app.py",
                "operation": "update_region",
                "reason": (
                    f"agent name {name!r} is already registered by {rendered_origins}; "
                    "choose a different slice name or extend the existing agent explicitly"
                ),
            }
        )
    unresolved_conflicts = [
        {
            "path": "app.py",
            "operation": "update_region",
            "reason": (
                "cannot determine the registered agent name for "
                f"{_render_agent_origin(origin)} without executing project code; "
                "use a literal name or extend the application manually"
            ),
        }
        for origin in sorted(agent_inspection.unresolved_origins)
    ]
    for precondition in agent_inspection.source_preconditions:
        _record_precondition(preconditions, precondition)

    tool_package_init = "tools/__init__.py"
    try:
        tool_package_path = _generated_path(root, tool_package_init)
    except _GeneratedPathError as exc:
        conflicts.append(
            {
                "path": tool_package_init,
                "operation": "create",
                "reason": str(exc),
            }
        )
    else:
        if not tool_package_path.exists():
            edits.append(_edit(tool_package_init, "create", ""))
        elif not tool_package_path.is_file():
            conflicts.append(
                {
                    "path": tool_package_init,
                    "operation": "create",
                    "reason": "path exists and is not a regular file",
                }
            )
        else:
            _record_precondition(
                preconditions,
                _file_precondition(root, tool_package_init),
            )

    for relative, content in sorted(independent.items()):
        try:
            path = _generated_path(root, relative)
        except _GeneratedPathError as exc:
            conflicts.append(
                {
                    "path": relative,
                    "operation": "create",
                    "reason": str(exc),
                }
            )
            continue
        if not path.exists():
            edits.append(_edit(relative, "create", content))
        elif not path.is_file():
            conflicts.append(
                {
                    "path": relative,
                    "operation": "create",
                    "reason": "path exists with user-authored or different content",
                }
            )
        else:
            existing_content = path.read_bytes()
            if existing_content != content.encode("utf-8"):
                conflicts.append(
                    {
                        "path": relative,
                        "operation": "create",
                        "reason": "path exists with user-authored or different content",
                    }
                )
            else:
                _record_precondition(
                    preconditions,
                    GeneratorPrecondition(
                        path=relative,
                        content_sha256=_sha256(existing_content),
                    ),
                )

    conflicts.extend(unresolved_conflicts)

    missing_anchors = [
        anchor
        for anchor in (
            GENERATED_IMPORTS_START,
            GENERATED_IMPORTS_END,
            GENERATED_REGISTRATIONS_START,
            GENERATED_REGISTRATIONS_END,
        )
        if app_source.count(anchor) != 1
    ]
    if missing_anchors:
        conflicts.append(
            {
                "path": "app.py",
                "operation": "update_region",
                "anchor": ", ".join(missing_anchors),
                "reason": "machine-owned registration anchors are missing or duplicated",
            }
        )
        return GeneratorPlan(
            status="manual_action_required",
            slice_name=name,
            tool_name=tool_name,
            effect=effect,
            edits=tuple(sorted(edits, key=lambda item: item.path)),
            preconditions=tuple(preconditions[path] for path in sorted(preconditions)),
            conflicts=tuple(conflicts),
            verification_commands=verification,
        )

    updated = _update_region(
        app_source,
        start=GENERATED_IMPORTS_START,
        end=GENERATED_IMPORTS_END,
        additions=import_lines,
    )
    updated = _update_region(
        updated,
        start=GENERATED_REGISTRATIONS_START,
        end=GENERATED_REGISTRATIONS_END,
        additions=[registration],
    )
    if updated != app_source:
        edits.append(
            _edit(
                "app.py",
                "update_region",
                updated,
                anchor=(f"{GENERATED_IMPORTS_START}; {GENERATED_REGISTRATIONS_START}"),
                preimage=app_source,
            )
        )
    else:
        _record_precondition(preconditions, app_precondition)
    if conflicts:
        status: Literal["ready", "conflict", "manual_action_required", "already_present"] = (
            "manual_action_required"
            if unresolved_conflicts and len(conflicts) == len(unresolved_conflicts)
            else "conflict"
        )
    elif not edits:
        status = "already_present"
    else:
        status = "ready"
    return GeneratorPlan(
        status=status,
        slice_name=name,
        tool_name=tool_name,
        effect=effect,
        edits=tuple(sorted(edits, key=lambda item: item.path)),
        preconditions=tuple(preconditions[path] for path in sorted(preconditions)),
        conflicts=tuple(conflicts),
        verification_commands=verification,
    )


def _inspect_registered_agents(
    root: Path,
    app_source: str,
) -> _AgentRegistrationInspection:
    """Inspect registered agent identities without importing or executing project code."""

    try:
        app_tree = ast.parse(app_source, filename="app.py")
    except SyntaxError as exc:
        raise ValueError(
            f"Cannot inspect registered agent identities in app.py: {exc.msg}."
        ) from exc
    parents = _ast_parents(app_tree)

    agent_import_candidates: dict[str, list[tuple[str, str]]] = {}
    for node in app_tree.body:
        if not isinstance(node, ast.ImportFrom) or node.module is None:
            continue
        if not node.module.startswith("agents."):
            continue
        for imported in node.names:
            local_name = imported.asname or imported.name
            agent_import_candidates.setdefault(local_name, []).append((node.module, imported.name))
    agent_imports = {
        local_name: origins[0]
        for local_name, origins in agent_import_candidates.items()
        if len(origins) == 1 and _module_binding_count(app_tree, local_name) == 1
    }

    registrations: dict[str, list[tuple[str, str]]] = {}
    source_preconditions: dict[str, GeneratorPrecondition] = {}
    module_snapshots: dict[str, _AgentModuleSnapshot | None] = {}
    unresolved: set[tuple[str, str]] = set()
    app_literals = _literal_string_bindings(app_tree)
    registration_aliases = _registration_aliases(app_tree)
    for node in ast.walk(app_tree):
        if not _is_agent_registration(node, registration_aliases):
            continue
        assert isinstance(node, ast.Call)
        registered = _registered_agent_argument(node)
        if registered is None:
            unresolved.add(("app.py", "register_agent"))
            continue
        if isinstance(registered, ast.Call):
            origin = ("app.py", "inline AgentSpec")
            agent_name = _literal_agent_spec_name(registered, app_literals)
            if agent_name is None:
                unresolved.add(origin)
            else:
                registrations.setdefault(agent_name, []).append(origin)
            continue
        if not isinstance(registered, ast.Name):
            unresolved.add(("app.py", "register_agent"))
            continue
        if _is_shadowed_in_enclosing_scope(
            node,
            registered.id,
            parents=parents,
        ):
            unresolved.add(("app.py", registered.id))
            continue
        origin = agent_imports.get(registered.id)
        if origin is None:
            origin = ("app.py", registered.id)
            expression = _assigned_expression(app_tree, registered.id)
            agent_name = _literal_agent_spec_name(expression, app_literals)
            if agent_name is None:
                unresolved.add(origin)
            else:
                registrations.setdefault(agent_name, []).append(origin)
            continue
        agent_name, source_precondition = _literal_agent_name(
            root,
            *origin,
            module_snapshots=module_snapshots,
        )
        if source_precondition is not None:
            source_preconditions[source_precondition.path] = source_precondition
        if agent_name is None:
            unresolved.add(origin)
        else:
            registrations.setdefault(agent_name, []).append(origin)
    return _AgentRegistrationInspection(
        origins_by_name={name: tuple(origins) for name, origins in registrations.items()},
        source_preconditions=tuple(
            source_preconditions[path] for path in sorted(source_preconditions)
        ),
        unresolved_origins=frozenset(unresolved),
    )


def _registration_aliases(tree: ast.Module) -> frozenset[str]:
    aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            value = node.value
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            value = node.value
            targets = [node.target]
        else:
            continue
        if not (isinstance(value, ast.Attribute) and value.attr == "register_agent"):
            continue
        aliases.update(target.id for target in targets if isinstance(target, ast.Name))
    return frozenset(aliases)


def _ast_parents(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    return {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}


def _is_shadowed_in_enclosing_scope(
    node: ast.AST,
    symbol: str,
    *,
    parents: dict[ast.AST, ast.AST],
) -> bool:
    current = parents.get(node)
    while current is not None:
        if _scope_binds_name(current, symbol):
            return True
        current = parents.get(current)
    return False


def _scope_binds_name(scope: ast.AST, symbol: str) -> bool:
    counter = _ModuleBindingCounter(symbol)
    if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
        if any(argument.arg == symbol for argument in _function_arguments(scope.args)):
            return True
        for statement in scope.body:
            counter.visit(statement)
        return counter.count > 0
    if isinstance(scope, ast.Lambda):
        if any(argument.arg == symbol for argument in _function_arguments(scope.args)):
            return True
        counter.visit(scope.body)
        return counter.count > 0
    if isinstance(scope, ast.ClassDef):
        for statement in scope.body:
            counter.visit(statement)
        return counter.count > 0
    if isinstance(scope, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
        return any(_target_binds_name(item.target, symbol) for item in scope.generators)
    return False


def _function_arguments(arguments: ast.arguments) -> tuple[ast.arg, ...]:
    positional = (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs)
    variadic = tuple(
        argument for argument in (arguments.vararg, arguments.kwarg) if argument is not None
    )
    return (*positional, *variadic)


def _is_agent_registration(node: ast.AST, aliases: frozenset[str]) -> bool:
    if not isinstance(node, ast.Call):
        return False
    return (isinstance(node.func, ast.Attribute) and node.func.attr == "register_agent") or (
        isinstance(node.func, ast.Name) and node.func.id in aliases
    )


def _registered_agent_argument(node: ast.Call) -> ast.expr | None:
    if node.args:
        return node.args[0]
    return next((item.value for item in node.keywords if item.arg == "spec"), None)


def _literal_agent_name(
    root: Path,
    module: str,
    symbol: str,
    *,
    module_snapshots: dict[str, _AgentModuleSnapshot | None],
) -> tuple[str | None, GeneratorPrecondition | None]:
    if module not in module_snapshots:
        relative = f"{module.replace('.', '/')}.py"
        try:
            path = _generated_path(root, relative)
        except _GeneratedPathError:
            module_snapshots[module] = None
        else:
            if not path.is_file():
                module_snapshots[module] = None
            else:
                content = path.read_bytes()
                try:
                    tree = ast.parse(content.decode("utf-8"), filename=relative)
                except SyntaxError as exc:
                    raise ValueError(
                        f"Cannot inspect registered agent identity in {relative}: {exc.msg}."
                    ) from exc
                module_snapshots[module] = _AgentModuleSnapshot(
                    tree=tree,
                    precondition=GeneratorPrecondition(
                        path=relative,
                        content_sha256=_sha256(content),
                    ),
                )
    snapshot = module_snapshots[module]
    if snapshot is None:
        return None, None
    value = _assigned_expression(snapshot.tree, symbol)
    return (
        _literal_agent_spec_name(value, _literal_string_bindings(snapshot.tree)),
        snapshot.precondition,
    )


def _assigned_expression(tree: ast.Module, symbol: str) -> ast.expr | None:
    candidates: list[ast.expr] = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == symbol for target in node.targets
        ):
            candidates.append(node.value)
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == symbol
            and node.value is not None
        ):
            candidates.append(node.value)
    if len(candidates) != 1 or _module_binding_count(tree, symbol) != 1:
        return None
    return candidates[0]


class _ModuleBindingCounter(ast.NodeVisitor):
    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        self.count = 0

    def visit_Assign(self, node: ast.Assign) -> None:
        self.count += sum(_target_binds_name(target, self.symbol) for target in node.targets)
        self.visit(node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self.count += _target_binds_name(node.target, self.symbol)
        if node.value is not None:
            self.visit(node.value)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.count += _target_binds_name(node.target, self.symbol)
        self.visit(node.value)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self.count += _target_binds_name(node.target, self.symbol)
        self.visit(node.value)

    def visit_Import(self, node: ast.Import) -> None:
        for imported in node.names:
            bound_name = imported.asname or imported.name.split(".", 1)[0]
            self.count += bound_name == self.symbol

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for imported in node.names:
            bound_name = imported.asname or imported.name
            self.count += bound_name == self.symbol

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.count += node.name == self.symbol

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.count += node.name == self.symbol

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.count += node.name == self.symbol

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        self.count += node.name == self.symbol
        for statement in node.body:
            self.visit(statement)

    def visit_MatchAs(self, node: ast.MatchAs) -> None:
        self.count += node.name == self.symbol
        if node.pattern is not None:
            self.visit(node.pattern)

    def visit_MatchStar(self, node: ast.MatchStar) -> None:
        self.count += node.name == self.symbol

    def visit_MatchMapping(self, node: ast.MatchMapping) -> None:
        self.count += node.rest == self.symbol
        for pattern in node.patterns:
            self.visit(pattern)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, (ast.Store, ast.Del)) and node.id == self.symbol:
            self.count += 1


def _module_binding_count(tree: ast.Module, symbol: str) -> int:
    counter = _ModuleBindingCounter(symbol)
    for node in tree.body:
        counter.visit(node)
    return counter.count


def _target_binds_name(target: ast.expr, symbol: str) -> bool:
    if isinstance(target, ast.Name):
        return target.id == symbol
    if isinstance(target, ast.Starred):
        return _target_binds_name(target.value, symbol)
    if isinstance(target, (ast.List, ast.Tuple)):
        return any(_target_binds_name(item, symbol) for item in target.elts)
    return False


def _literal_string_bindings(tree: ast.Module) -> dict[str, str]:
    bindings: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        value = _assigned_expression(tree, target.id)
        if isinstance(value, ast.Constant) and type(value.value) is str:
            bindings[target.id] = value.value
    return bindings


def _literal_agent_spec_name(
    value: ast.expr | None,
    string_bindings: dict[str, str],
) -> str | None:
    if not isinstance(value, ast.Call):
        return None
    constructor = value.func
    if not (
        (isinstance(constructor, ast.Name) and constructor.id == "AgentSpec")
        or (isinstance(constructor, ast.Attribute) and constructor.attr == "AgentSpec")
    ):
        return None
    for keyword_argument in value.keywords:
        if keyword_argument.arg != "name":
            continue
        name_value = keyword_argument.value
        if isinstance(name_value, ast.Constant) and type(name_value.value) is str:
            return name_value.value
        if isinstance(name_value, ast.Name):
            return string_bindings.get(name_value.id)
    return None


def _render_agent_origin(origin: tuple[str, str]) -> str:
    if origin[0] == "app.py":
        return f"app.py:{origin[1]}"
    return f"{origin[0]}.{origin[1]}"


def _region_contains_statement(
    source: str,
    *,
    start: str,
    end: str,
    statement: str,
) -> bool:
    bounds = _region_bounds(source, start=start, end=end)
    if bounds is None:
        return False
    body_start, body_end, _ = bounds
    existing, _ = _parse_region_statements(source[body_start:body_end])
    return _statement_key(statement) in {item.key for item in existing}


def apply_slice_plan(plan: GeneratorPlan) -> None:
    """Apply a ready slice plan as an all-or-nothing filesystem transaction."""

    if plan.status != "ready":
        raise GeneratorApplyError(f"Only ready generator plans can be applied, not {plan.status}.")
    project = resolve_project(command="cayu generate")
    root = project.root.resolve()
    targets: list[tuple[GeneratorEdit, Path]] = []
    seen: set[Path] = set()
    try:
        _validate_plan_preconditions(plan, root)
        for edit in plan.edits:
            target = _generated_path(root, edit.path)
            if target in seen:
                raise GeneratorApplyError(f"Generator plan contains duplicate path: {edit.path}")
            seen.add(target)
            _validate_edit_preimage(edit, target)
            targets.append((edit, target))
    except (_GeneratedPathError, OSError) as exc:
        raise GeneratorApplyError(str(exc)) from exc

    stage_root = Path(tempfile.mkdtemp(prefix=".cayu-generate-", dir=root))
    staged: dict[str, Path] = {}
    backups: dict[str, Path] = {}
    applied: list[tuple[GeneratorEdit, Path]] = []
    created_directories: list[Path] = []
    try:
        for index, (edit, target) in enumerate(targets):
            staged_path = stage_root / f"edit-{index}"
            staged_path.write_bytes(edit.content.encode("utf-8"))
            if edit.operation == "update_region":
                staged_path.chmod(target.stat().st_mode)
                backup_path = stage_root / f"backup-{index}"
                backup_path.write_bytes(target.read_bytes())
                backup_path.chmod(target.stat().st_mode)
                backups[edit.path] = backup_path
            staged[edit.path] = staged_path

        for edit, target in targets:
            try:
                _validate_plan_preconditions(plan, root)
                _generated_path(root, edit.path)
                _validate_edit_preimage(edit, target)
                _create_missing_parents(root, target.parent, created_directories)
                staged[edit.path].replace(target)
                applied.append((edit, target))
            except (GeneratorApplyError, _GeneratedPathError, OSError) as exc:
                raise GeneratorApplyError(str(exc)) from exc
    except Exception as exc:
        rollback_errors: list[str] = []
        for edit, target in reversed(applied):
            try:
                if edit.operation == "create":
                    target.unlink()
                else:
                    backups[edit.path].replace(target)
            except OSError as rollback_exc:
                rollback_errors.append(f"{edit.path}: {rollback_exc}")
        for directory in reversed(created_directories):
            with suppress(OSError):
                directory.rmdir()
        if rollback_errors:
            details = "; ".join(rollback_errors)
            raise GeneratorApplyError(
                f"{exc}; rollback could not restore every path: {details}"
            ) from exc
        if isinstance(exc, GeneratorApplyError):
            raise
        raise GeneratorApplyError(str(exc)) from exc
    finally:
        shutil.rmtree(stage_root, ignore_errors=True)


def _validate_plan_preconditions(plan: GeneratorPlan, root: Path) -> None:
    seen: set[Path] = set()
    for precondition in plan.preconditions:
        path = _generated_path(root, precondition.path)
        if path in seen:
            raise GeneratorApplyError(
                f"Generator plan contains duplicate precondition: {precondition.path}"
            )
        seen.add(path)
        if not path.is_file() or _sha256(path.read_bytes()) != precondition.content_sha256:
            raise GeneratorApplyError(f"{precondition.path} changed after the plan was created.")


def _validate_edit_preimage(edit: GeneratorEdit, target: Path) -> None:
    content_sha256 = _sha256(edit.content.encode("utf-8"))
    if content_sha256 != edit.content_sha256:
        raise GeneratorApplyError(f"Planned content hash does not match for {edit.path}.")
    if edit.operation == "create":
        if edit.preimage_sha256 is not None:
            raise GeneratorApplyError(f"Create edit has an unexpected preimage for {edit.path}.")
        if target.exists() or target.is_symlink():
            raise GeneratorApplyError(f"{edit.path} changed after the plan was created.")
        return
    if edit.preimage_sha256 is None:
        raise GeneratorApplyError(f"Update edit is missing its preimage hash for {edit.path}.")
    if not target.is_file() or _sha256(target.read_bytes()) != edit.preimage_sha256:
        raise GeneratorApplyError(f"{edit.path} changed after the plan was created.")


def _create_missing_parents(root: Path, parent: Path, created: list[Path]) -> None:
    missing: list[Path] = []
    current = parent
    while current != root and not current.exists():
        missing.append(current)
        current = current.parent
    if current != root:
        try:
            current.relative_to(root)
        except ValueError as exc:
            raise GeneratorApplyError(
                f"Generated parent escapes the project root: {current}"
            ) from exc
    if current.is_symlink() or not current.is_dir():
        raise GeneratorApplyError(f"Generated parent is not a real directory: {current}")
    for directory in reversed(missing):
        directory.mkdir()
        created.append(directory)


def _generated_path(root: Path, relative: str) -> Path:
    relative_path = Path(relative)
    if relative_path.is_absolute() or not relative_path.parts or ".." in relative_path.parts:
        raise _GeneratedPathError(f"generated path escapes the project root: {relative}")
    target = root.joinpath(relative_path)
    current = root
    for part in relative_path.parts:
        current = current / part
        if current.is_symlink():
            prefix = current.relative_to(root).as_posix()
            raise _GeneratedPathError(f"generated path contains a symbolic link: {prefix}")
    try:
        target.resolve(strict=False).relative_to(root.resolve())
    except ValueError as exc:
        raise _GeneratedPathError(f"generated path escapes the project root: {relative}") from exc
    return target


def _slice_files(*, name: str, tool_name: str, effect: str) -> dict[str, str]:
    agent_constant = f"{_constant_name(name)}_AGENT"
    tool_name_constant = f"{_constant_name(tool_name)}_TOOL_NAME"
    tool_class = f"{_class_name(tool_name)}Tool"
    effect_constant = effect.upper()
    if effect == "external":
        test_assertions = """    assert outcome.status is SessionStatus.INTERRUPTED
    assert any(
        event.type == EventType.TOOL_CALL_APPROVAL_REQUESTED for event in outcome.events
    )"""
        eval_assertions = """                        SessionInterrupted(),
                        EventOccurred(EventType.TOOL_CALL_APPROVAL_REQUESTED),"""
        test_effect_imports = {"EventType", "SessionStatus"}
        eval_effect_imports = {"EventOccurred", "EventType", "SessionInterrupted"}
    else:
        test_assertions = f'''    assert outcome.ok
    assert outcome.final_text == "{name} completed sample."'''
        eval_assertions = f"""                        SessionCompleted(),
                        ToolCalled({tool_name_constant}),
                        FinalOutputContains("sample"),"""
        test_effect_imports = set()
        eval_effect_imports = {"FinalOutputContains", "SessionCompleted", "ToolCalled"}
    test_imports = "\n".join(
        f"    {import_name},"
        for import_name in sorted(
            {
                "InMemorySessionStore",
                "InMemoryTaskStore",
                "Message",
                "ModelStreamEvent",
                "RunRequest",
                "ScriptedModelProvider",
                "run_to_completion",
                *test_effect_imports,
            }
        )
    )
    eval_imports = "\n".join(
        f"    {import_name},"
        for import_name in sorted(
            {
                "EvalCase",
                "EvalPlan",
                "EvalSuite",
                "InMemorySessionStore",
                "InMemoryTaskStore",
                "Message",
                "ModelStreamEvent",
                "RunRequest",
                "ScriptedModelProvider",
                *eval_effect_imports,
            }
        )
    )
    agent = f'''from cayu import AgentAuthoringState, AgentSpec

from tools.{tool_name} import {tool_name_constant}


{agent_constant} = AgentSpec(
    name="{name}",
    model="gpt-5.6-luna",
    system_prompt=f"Use {{{tool_name_constant}}} when it directly answers the user's request.",
    workflow_tool_names=({tool_name_constant},),
    authoring_state=AgentAuthoringState.UNFINISHED_GENERATED_TRACER_BULLET,
)
'''
    tool = f'''from cayu import Tool, ToolContext, ToolEffect, ToolResult, ToolSpec


{tool_name_constant} = "{tool_name}"


class {tool_class}(Tool):
    spec = ToolSpec(
        name={tool_name_constant},
        effect=ToolEffect.{effect_constant},
        description="Process one explicit input for the {name} agent.",
        input_schema={{
            "type": "object",
            "properties": {{"input": {{"type": "string"}}}},
            "required": ["input"],
            "additionalProperties": False,
        }},
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        return ToolResult(content=f"{tool_name}: {{args['input']}}")
'''
    test = f'''from __future__ import annotations

import asyncio

from cayu import (
{test_imports}
)

from app import build_app
from tools.{tool_name} import {tool_name_constant}


def test_{name}_slice_runs_through_public_runtime_seams() -> None:
    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    name={tool_name_constant}, arguments={{"input": "sample"}}
                ),
                ModelStreamEvent.completed({{"finish_reason": "tool_calls"}}),
            ],
            [
                ModelStreamEvent.text_delta("{name} completed sample."),
                ModelStreamEvent.completed({{"finish_reason": "stop"}}),
            ],
        ]
    )
    app = build_app(
        provider=provider,
        session_store=InMemorySessionStore(),
        task_store=InMemoryTaskStore(),
    )
    outcome = asyncio.run(
        run_to_completion(
            app,
            RunRequest(
                agent_name="{name}",
                messages=[Message.text("user", "Process sample")],
                max_steps=2,
            ),
        )
    )
{test_assertions}
'''
    eval_source = f'''from cayu import (
{eval_imports}
)

from app import build_app
from tools.{tool_name} import {tool_name_constant}


def build_eval() -> EvalPlan:
    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    name={tool_name_constant}, arguments={{"input": "sample"}}
                ),
                ModelStreamEvent.completed({{"finish_reason": "tool_calls"}}),
            ],
            [
                ModelStreamEvent.text_delta("{name} completed sample."),
                ModelStreamEvent.completed({{"finish_reason": "stop"}}),
            ],
        ]
    )
    app = build_app(
        provider=provider,
        session_store=InMemorySessionStore(),
        task_store=InMemoryTaskStore(),
    )
    return EvalPlan(
        app=app,
        suite=EvalSuite(
            id="{name}-trajectory",
            cases=[
                EvalCase(
                    id="{name}-uses-{tool_name}",
                    request=RunRequest(
                        agent_name="{name}",
                        messages=[Message.text("user", "Process sample")],
                        max_steps=2,
                    ),
                    assertions=[
{eval_assertions}
                    ],
                )
            ],
        ),
    )
'''
    return {
        f"agents/{name}.py": agent,
        f"tools/{tool_name}.py": tool,
        f"tests/test_{name}.py": test,
        f"evals/{name}.py": eval_source,
    }


def _edit(
    path: str,
    operation: Literal["create", "update_region"],
    content: str,
    *,
    anchor: str | None = None,
    preimage: str | None = None,
) -> GeneratorEdit:
    return GeneratorEdit(
        path=path,
        operation=operation,
        content=content,
        content_sha256=_sha256(content.encode("utf-8")),
        preimage_sha256=None if preimage is None else _sha256(preimage.encode("utf-8")),
        anchor=anchor,
    )


def _file_precondition(root: Path, relative: str) -> GeneratorPrecondition:
    path = _generated_path(root, relative)
    if not path.is_file():
        raise ValueError(f"Generator precondition is not a regular file: {relative}.")
    return GeneratorPrecondition(
        path=relative,
        content_sha256=_sha256(path.read_bytes()),
    )


def _record_precondition(
    preconditions: dict[str, GeneratorPrecondition],
    precondition: GeneratorPrecondition,
) -> None:
    existing = preconditions.get(precondition.path)
    if existing is not None and existing.content_sha256 != precondition.content_sha256:
        raise ValueError(f"{precondition.path} changed while the generator plan was being created.")
    preconditions[precondition.path] = precondition


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _update_region(source: str, *, start: str, end: str, additions: list[str]) -> str:
    bounds = _region_bounds(source, start=start, end=end)
    if bounds is None:
        raise ValueError(f"Generated region is malformed: {start}; {end}.")
    body_start, body_end, indent = bounds
    existing, trailing = _parse_region_statements(source[body_start:body_end])
    existing_keys = {item.key for item in existing}
    missing: list[_RegionStatement] = []
    for addition in additions:
        key = _statement_key(addition)
        if key not in existing_keys:
            missing.append(_RegionStatement(key=key, source=addition))
            existing_keys.add(key)
    if not missing:
        return source

    statements = sorted((*existing, *missing), key=lambda item: (item.key, item.source))
    rendered: list[str] = []
    for item in statements:
        statement_source = item.source.strip("\n")
        if statement_source:
            rendered.append(textwrap.indent(statement_source, indent) + "\n")
    if trailing.strip():
        rendered.append(textwrap.indent(trailing.strip("\n"), indent) + "\n")
    return source[:body_start] + "".join(rendered) + source[body_end:]


def _region_bounds(
    source: str,
    *,
    start: str,
    end: str,
) -> tuple[int, int, str] | None:
    if source.count(start) != 1 or source.count(end) != 1:
        return None
    start_index = source.index(start)
    end_index = source.index(end)
    if end_index <= start_index:
        return None
    try:
        body_start = source.index("\n", start_index) + 1
    except ValueError:
        return None
    body_end = source.rfind("\n", 0, end_index) + 1
    if body_end < body_start:
        return None
    line_start = source.rfind("\n", 0, start_index) + 1
    indent = source[line_start:start_index]
    if indent.strip():
        return None
    return body_start, body_end, indent


def _parse_region_statements(
    body: str,
) -> tuple[tuple[_RegionStatement, ...], str]:
    dedented = textwrap.dedent(body)
    try:
        tree = ast.parse(dedented or "\n")
    except SyntaxError as exc:
        raise ValueError(f"Cannot parse generated region: {exc.msg}.") from exc
    lines = dedented.splitlines(keepends=True)
    statements: list[_RegionStatement] = []
    cursor = 0
    for node in tree.body:
        start_line = node.lineno - 1
        end_line = node.end_lineno or node.lineno
        if start_line < cursor or end_line <= start_line:
            raise ValueError("Cannot determine generated statement boundaries.")
        statement_source = "".join(lines[cursor:end_line])
        statements.append(
            _RegionStatement(
                key=ast.dump(node, include_attributes=False),
                source=statement_source,
            )
        )
        cursor = end_line
    return tuple(statements), "".join(lines[cursor:])


def _statement_key(source: str) -> str:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ValueError(f"Cannot parse generated statement: {exc.msg}.") from exc
    if len(tree.body) != 1:
        raise ValueError("Generated region additions must contain exactly one statement.")
    return ast.dump(tree.body[0], include_attributes=False)


def _identifier(value: str, label: str) -> str:
    if not _IDENTIFIER_RE.fullmatch(value):
        raise ValueError(f"{label} must be a snake_case Python identifier.")
    if keyword.iskeyword(value):
        raise ValueError(f"{label} must not be a Python keyword: {value}.")
    return value


def _constant_name(value: str) -> str:
    return value.upper()


def _class_name(value: str) -> str:
    return "".join(part.capitalize() for part in value.split("_"))


def _render_plan(plan: GeneratorPlan, *, applied: bool) -> str:
    action = "Applied" if applied else "Planned"
    lines = [f"{action} {plan.slice_name}: {plan.status}"]
    lines.extend(f"  {edit.operation}: {edit.path}" for edit in plan.edits)
    lines.extend(f"  conflict: {item['path']} — {item['reason']}" for item in plan.conflicts)
    if applied:
        lines.append("Verify:")
        lines.extend(f"  {command}" for command in plan.verification_commands)
    return "\n".join(lines)
