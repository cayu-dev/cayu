from __future__ import annotations

import argparse
import ast
import functools
import hashlib
import json
import keyword
import re
import sys
import textwrap
import tomllib
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, ParamSpec, TypeVar

from pydantic import BaseModel, ConfigDict

from cayu.cli._generator_transaction import (
    GeneratorTransactionEdit,
    GeneratorTransactionError,
    GeneratorTransactionPrecondition,
    GeneratorTransactionRequest,
    apply_generator_transaction,
    encode_generator_transaction_content,
    generator_planning_guard,
    generator_transaction_staged_byte_limit,
    recover_generator_transaction,
    validate_generator_transaction_collection_bounds,
)
from cayu.cli._output import add_output_options, output_destination
from cayu.cli.project import ProjectError, resolve_project
from cayu.cli.scaffold import (
    GENERATED_AGENT_CONFIG_END,
    GENERATED_AGENT_CONFIG_START,
    GENERATED_AGENT_IMPORTS_END,
    GENERATED_AGENT_IMPORTS_START,
    GENERATED_IMPORTS_END,
    GENERATED_IMPORTS_START,
    GENERATED_REGISTRATIONS_END,
    GENERATED_REGISTRATIONS_START,
    GENERATED_STARTER_TOOLS_END,
    GENERATED_STARTER_TOOLS_START,
    PROVIDER_OVERRIDE_AGENT_HELPER,
)
from cayu.core.agents import AgentAuthoringState
from cayu.runtime.manifest import APP_MANIFEST_SCHEMA_VERSION

_IDENTIFIER_RE = re.compile(r"[a-z][a-z0-9_]*")
GENERATOR_PLAN_SCHEMA_VERSION = APP_MANIFEST_SCHEMA_VERSION
_P = ParamSpec("_P")
_PlanT = TypeVar("_PlanT", bound="GeneratorPlan | ServiceContextMigrationPlan")


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

    schema_version: Literal["16"] = GENERATOR_PLAN_SCHEMA_VERSION
    status: Literal["ready", "conflict", "manual_action_required", "already_present"]
    slice_name: str
    tool_name: str
    effect: str
    authoring_state: AgentAuthoringState = AgentAuthoringState.UNFINISHED_GENERATED_TRACER_BULLET
    edits: tuple[GeneratorEdit, ...]
    preconditions: tuple[GeneratorPrecondition, ...] = ()
    conflicts: tuple[dict[str, str], ...] = ()
    verification_commands: tuple[str, ...]


class ServiceContextMigrationPlan(BaseModel):
    """One reviewable migration for a generated maintained-service factory."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["16"] = GENERATOR_PLAN_SCHEMA_VERSION
    status: Literal["ready", "conflict", "manual_action_required", "already_present"]
    migration: Literal["service_context"] = "service_context"
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


def _guarded_generator_plan(
    function: Callable[_P, _PlanT],
) -> Callable[_P, _PlanT]:
    @functools.wraps(function)
    def guarded(*args: _P.args, **kwargs: _P.kwargs) -> _PlanT:
        project = resolve_project(command="cayu generate")
        with generator_planning_guard(project.root):
            return function(*args, **kwargs)

    return guarded


def add_generate_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "generate",
        help="Plan or add reviewable generated Cayu application slices.",
        description=(
            "Plan or add reviewable generated Cayu application slices. "
            "Use `--dry-run` to inspect the plan without writing files."
        ),
    )
    generators = parser.add_subparsers(dest="generate_command", required=True)
    slice_parser = generators.add_parser(
        "slice",
        help="Add one agent, typed tool, runtime test, and trajectory eval.",
        description=(
            "Add one agent, typed tool, runtime test, and trajectory eval. "
            "Finish the generated tracer bullet, then run its verification commands."
        ),
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
    add_output_options(slice_parser)
    tool_parser = generators.add_parser(
        "tool",
        help="Attach the first generated tool tracer bullet to the starter agent.",
        description=(
            "Attach the first generated tool tracer bullet to the starter agent. "
            "Finish its domain behavior, then run `cayu check`."
        ),
    )
    tool_parser.add_argument("name", help="snake_case tool name.")
    tool_parser.add_argument(
        "--agent",
        required=True,
        help="Existing scaffolded starter agent name.",
    )
    tool_parser.add_argument(
        "--effect",
        choices=("none", "idempotent", "external"),
        required=True,
        help="Declared ToolEffect. See `cayu guide tool-effects` for the decision table.",
    )
    tool_parser.add_argument("--dry-run", action="store_true", help="Plan without writes.")
    add_output_options(tool_parser)
    service_context_parser = generators.add_parser(
        "service-context",
        help="Migrate a generated maintained service to Cayu project context assembly.",
        description=(
            "Safely add the framework-owned project_context pass-through required for "
            "zero-code Control Plane features. Customized factories receive manual guidance."
        ),
    )
    service_context_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan without writes.",
    )
    add_output_options(service_context_parser)


def run_generate(args: argparse.Namespace) -> int:
    try:
        with output_destination(args.output):
            return _run_generate(args)
    except OSError as exc:
        print(f"error: could not write output: {exc}", file=sys.stderr)
        return 2


def _run_generate(args: argparse.Namespace) -> int:
    if args.generate_command not in {"slice", "tool", "service-context"}:
        return 2
    try:
        project = resolve_project(command="cayu generate")
        recover_generator_transaction(project.root, dry_run=args.dry_run)
        if args.generate_command == "slice":
            plan = plan_slice(
                name=args.name,
                tool_name=args.tool,
                effect=args.effect,
            )
        elif args.generate_command == "tool":
            plan = plan_tool(
                tool_name=args.name,
                agent_name=args.agent,
                effect=args.effect,
            )
        else:
            plan = plan_service_context()
    except (GeneratorTransactionError, ProjectError, ValueError, OSError) as exc:
        if args.output_format == "json":
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

    should_apply = not args.dry_run and plan.status == "ready"
    if should_apply:
        try:
            if isinstance(plan, ServiceContextMigrationPlan):
                apply_service_context_plan(plan)
            else:
                apply_slice_plan(plan)
        except (GeneratorApplyError, ProjectError, OSError) as exc:
            if args.output_format == "json":
                print(
                    json.dumps(
                        {
                            "schema_version": GENERATOR_PLAN_SCHEMA_VERSION,
                            "error": {"code": "GENERATOR_APPLY_FAILED", "message": str(exc)},
                        },
                        sort_keys=True,
                    )
                )
            else:
                print(f"error: {exc}", file=sys.stderr)
            return 2
    if args.output_format == "json":
        print(plan.model_dump_json(indent=2))
    else:
        print(
            _render_service_context_plan(plan, applied=should_apply)
            if isinstance(plan, ServiceContextMigrationPlan)
            else _render_plan(plan, applied=should_apply)
        )
    return 0 if plan.status in {"ready", "already_present"} else 1


@_guarded_generator_plan
def plan_service_context() -> ServiceContextMigrationPlan:
    """Plan the narrow generated-service migration without rewriting user code."""

    verification = (
        "uv run --no-sync cayu check --json",
        "uv run --no-sync pytest -q tests/test_public_service_security.py",
    )
    project = resolve_project(command="cayu generate service-context")
    if project.service_target != "service:build_service":
        return ServiceContextMigrationPlan(
            status="manual_action_required",
            edits=(),
            conflicts=(
                {
                    "path": "pyproject.toml",
                    "operation": "update_region",
                    "reason": (
                        "automatic migration supports the generated "
                        'service_factory = "service:build_service" contract only'
                    ),
                },
            ),
            verification_commands=verification,
        )
    service_path = _generated_path(project.root, "service.py")
    if not service_path.is_file():
        return ServiceContextMigrationPlan(
            status="manual_action_required",
            edits=(),
            conflicts=(
                {
                    "path": "service.py",
                    "operation": "update_region",
                    "reason": "the generated maintained-service module is missing",
                },
            ),
            verification_commands=verification,
        )
    source = service_path.read_text(encoding="utf-8")
    try:
        updated = _service_context_migration_source(source)
    except (SyntaxError, ValueError) as exc:
        return ServiceContextMigrationPlan(
            status="manual_action_required",
            edits=(),
            preconditions=(_file_precondition(project.root, "service.py"),),
            conflicts=(
                {
                    "path": "service.py",
                    "operation": "update_region",
                    "reason": str(exc),
                },
            ),
            verification_commands=verification,
        )
    if updated == source:
        return ServiceContextMigrationPlan(
            status="already_present",
            edits=(),
            preconditions=(_file_precondition(project.root, "service.py"),),
            verification_commands=verification,
        )
    return ServiceContextMigrationPlan(
        status="ready",
        edits=(
            _edit(
                "service.py",
                "update_region",
                updated,
                anchor="build_service/project_context",
                preimage=source,
            ),
        ),
        verification_commands=verification,
    )


def _service_context_migration_source(source: str) -> str:
    tree = ast.parse(source, filename="service.py")
    server_imports = [
        node
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == "cayu.server"
    ]
    factories = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "build_service"
    ]
    if len(server_imports) != 1 or len(factories) != 1:
        raise ValueError("expected one top-level cayu.server import and one build_service function")
    server_import = server_imports[0]
    factory = factories[0]
    calls = [
        node
        for node in ast.walk(factory)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "create_agent_service"
    ]
    if len(calls) != 1:
        raise ValueError("expected one direct create_agent_service() call in build_service")
    call = calls[0]

    imported = any(
        alias.name == "ProjectControlPlaneContext" and alias.asname is None
        for alias in server_import.names
    )
    keyword_arguments = {argument.arg: argument for argument in factory.args.kwonlyargs}
    declared = "project_context" in keyword_arguments
    passed_keywords = [keyword for keyword in call.keywords if keyword.arg == "project_context"]
    passed = (
        len(passed_keywords) == 1
        and isinstance(passed_keywords[0].value, ast.Name)
        and passed_keywords[0].value.id == "project_context"
    )
    if imported and declared and passed:
        return source
    if passed_keywords and not passed:
        raise ValueError("project_context is already passed with a customized value")
    if declared:
        argument = keyword_arguments["project_context"]
        index = factory.args.kwonlyargs.index(argument)
        default = factory.args.kw_defaults[index]
        if not isinstance(default, ast.Constant) or default.value is not None:
            raise ValueError("project_context must remain an optional keyword-only parameter")

    lines = source.splitlines(keepends=True)
    newline = "\r\n" if "\r\n" in source else "\n"
    insertions: list[tuple[int, str]] = []
    if not imported:
        if server_import.end_lineno is None or not server_import.names:
            raise ValueError("cayu.server import does not expose stable source positions")
        closing_index = server_import.end_lineno - 1
        if lines[closing_index].strip() != ")":
            raise ValueError("cayu.server imports must use the generated multiline form")
        indentation = " " * server_import.names[0].col_offset
        insertions.append((closing_index, f"{indentation}ProjectControlPlaneContext,{newline}"))
    if not declared:
        mode_arguments = [
            argument for argument in factory.args.kwonlyargs if argument.arg == "mode"
        ]
        if len(mode_arguments) != 1:
            raise ValueError("build_service must expose the generated keyword-only mode parameter")
        mode_argument = mode_arguments[0]
        mode_end_lineno = mode_argument.end_lineno
        if mode_end_lineno is None:
            raise ValueError("build_service mode parameter has no stable source position")
        mode_line = lines[mode_end_lineno - 1]
        if mode_line.strip() != "mode: ServiceMode,":
            raise ValueError("build_service mode parameter uses a customized source layout")
        indentation = " " * mode_argument.col_offset
        insertions.append(
            (
                mode_end_lineno,
                f"{indentation}project_context: ProjectControlPlaneContext | None = None,{newline}",
            )
        )
    if not passed:
        mode_keywords = [keyword for keyword in call.keywords if keyword.arg == "mode"]
        if len(mode_keywords) != 1:
            raise ValueError("create_agent_service must receive the generated mode keyword")
        mode_keyword = mode_keywords[0]
        mode_end_lineno = mode_keyword.end_lineno
        if mode_end_lineno is None:
            raise ValueError("create_agent_service mode keyword has no stable source position")
        mode_line = lines[mode_end_lineno - 1]
        if mode_line.strip() != "mode=mode,":
            raise ValueError("create_agent_service mode keyword uses a customized source layout")
        indentation = " " * mode_keyword.col_offset
        insertions.append(
            (mode_end_lineno, f"{indentation}project_context=project_context,{newline}")
        )
    for index, content in sorted(insertions, reverse=True):
        lines.insert(index, content)
    updated = "".join(lines)
    if not _service_context_contract_present(updated):
        raise ValueError("the proposed migration did not produce the complete context contract")
    return updated


def _service_context_contract_present(source: str) -> bool:
    try:
        tree = ast.parse(source, filename="service.py")
    except SyntaxError:
        return False
    imported = any(
        isinstance(node, ast.ImportFrom)
        and node.module == "cayu.server"
        and any(
            alias.name == "ProjectControlPlaneContext" and alias.asname is None
            for alias in node.names
        )
        for node in tree.body
    )
    factories = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "build_service"
    ]
    if not imported or len(factories) != 1:
        return False
    factory = factories[0]
    declared = any(argument.arg == "project_context" for argument in factory.args.kwonlyargs)
    passed = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "create_agent_service"
        and any(
            keyword.arg == "project_context"
            and isinstance(keyword.value, ast.Name)
            and keyword.value.id == "project_context"
            for keyword in node.keywords
        )
        for node in ast.walk(factory)
    )
    return declared and passed


def _registration_target(root: Path) -> tuple[str, Path]:
    """Select the declared convention seam or the explicit factory of a freeform project."""

    convention_relative = "agents/registration.py"
    convention_path = _generated_path(root, convention_relative)
    pyproject = _generated_path(root, "pyproject.toml")
    declared_convention = False
    if pyproject.is_file():
        try:
            document = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, tomllib.TOMLDecodeError):
            document = {}
        scaffold = document.get("tool", {}).get("cayu", {}).get("scaffold", {})
        declared_convention = scaffold.get("convention") == 1
    if declared_convention:
        return convention_relative, convention_path
    factory_relative = "app.py"
    return factory_relative, _generated_path(root, factory_relative)


@_guarded_generator_plan
def plan_tool(*, tool_name: str, agent_name: str, effect: str) -> GeneratorPlan:
    """Plan the first tool tracer bullet for the updated scaffold starter."""

    tool_name = _identifier(tool_name, "tool name")
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", agent_name):
        raise ValueError("agent name must match a scaffolded Cayu agent name.")
    if effect not in {"none", "idempotent", "external"}:
        raise ValueError("effect must be none, idempotent, or external.")
    project = resolve_project(command="cayu generate")
    root = project.root
    registration_relative, registration_path = _registration_target(root)
    agent_path = _generated_path(root, "agents/agent.py")
    if not registration_path.is_file() or not agent_path.is_file():
        raise ValueError(
            "First-tool generation requires the generated registration seam and "
            "agents/agent.py from `cayu new`."
        )
    app_source = registration_path.read_text(encoding="utf-8")
    agent_source = agent_path.read_text(encoding="utf-8")
    conflicts: list[dict[str, str]] = []
    edits: list[GeneratorEdit] = []
    preconditions: dict[str, GeneratorPrecondition] = {}
    marker_contracts = (
        (
            registration_relative,
            app_source,
            GENERATED_IMPORTS_START,
            GENERATED_IMPORTS_END,
        ),
        (
            registration_relative,
            app_source,
            GENERATED_STARTER_TOOLS_START,
            GENERATED_STARTER_TOOLS_END,
        ),
        (
            "agents/agent.py",
            agent_source,
            GENERATED_AGENT_IMPORTS_START,
            GENERATED_AGENT_IMPORTS_END,
        ),
        (
            "agents/agent.py",
            agent_source,
            GENERATED_AGENT_CONFIG_START,
            GENERATED_AGENT_CONFIG_END,
        ),
    )
    missing_markers = [
        f"{path}: {start}; {end}"
        for path, source, start, end in marker_contracts
        if source.count(start) != 1 or source.count(end) != 1
    ]
    if missing_markers:
        return GeneratorPlan(
            status="manual_action_required",
            slice_name=agent_name,
            tool_name=tool_name,
            effect=effect,
            edits=(),
            conflicts=(
                {
                    "path": f"{registration_relative} / agents/agent.py",
                    "operation": "update_region",
                    "reason": (
                        "first-tool generation requires intact machine-owned starter markers; "
                        f"missing or duplicated: {', '.join(missing_markers)}"
                    ),
                },
            ),
            verification_commands=_tool_verification_commands(tool_name),
        )

    inspection = _normalize_declared_registration_inspection(
        root,
        registration_relative,
        _inspect_registered_agents(
            root,
            app_source,
            source_path=registration_relative,
        ),
    )
    for precondition in inspection.source_preconditions:
        _record_precondition(preconditions, precondition)
    origins = inspection.origins_by_name.get(agent_name, ())
    expected_origin = ("agents.agent", "AGENT")
    if origins != (expected_origin,):
        rendered = ", ".join(_render_agent_origin(origin) for origin in origins) or "none"
        conflicts.append(
            {
                "path": registration_relative,
                "operation": "update_region",
                "reason": (
                    f"agent {agent_name!r} is not the scaffold starter registered from "
                    f"agents.agent.AGENT (found: {rendered})"
                ),
            }
        )
    if inspection.unresolved_origins:
        conflicts.append(
            {
                "path": registration_relative,
                "operation": "update_region",
                "reason": (
                    "cannot safely attach a first tool while an agent registration has a "
                    "dynamic identity"
                ),
            }
        )
    if not conflicts and effect == "external" and _declared_scaffold_preset(root) == "coding":
        return GeneratorPlan(
            status="manual_action_required",
            slice_name=agent_name,
            tool_name=tool_name,
            effect=effect,
            edits=(),
            preconditions=tuple(preconditions[path] for path in sorted(preconditions)),
            conflicts=(
                {
                    "path": registration_relative,
                    "operation": "update_region",
                    "reason": (
                        "the maintained coding primary has an existing constrained policy; "
                        "use `cayu generate slice NAME --tool TOOL --effect external` to "
                        "create an independent approval boundary"
                    ),
                },
            ),
            verification_commands=_tool_verification_commands(tool_name),
        )

    tool_class = f"{_class_name(tool_name)}Tool"
    tool_constant = f"{_constant_name(tool_name)}_TOOL_NAME"
    app_imports = [f"from tools.{tool_name} import {tool_class}"]
    app_tool_statements = [f"starter_tools.append({tool_class}())"]
    if effect == "external":
        app_imports = [f"from tools.{tool_name} import {tool_class}, {tool_constant}"]
        app_tool_statements.append(f"starter_external_tool_names.append({tool_constant})")
    agent_imports = [f"from tools.{tool_name} import {tool_constant}"]
    agent_config = [
        '_SYSTEM_PROMPT_PARTS.append("Use the generated tool when it directly answers '
        "the user's request.\")",
        f"_WORKFLOW_TOOL_NAMES.append({tool_constant})",
        '_AUTHORING_STATE = "unfinished_generated_tracer_bullet"',
    ]
    generated_regions = (
        (
            registration_relative,
            app_source,
            GENERATED_STARTER_TOOLS_START,
            GENERATED_STARTER_TOOLS_END,
            app_tool_statements,
        ),
        (
            "agents/agent.py",
            agent_source,
            GENERATED_AGENT_IMPORTS_START,
            GENERATED_AGENT_IMPORTS_END,
            agent_imports,
        ),
        (
            "agents/agent.py",
            agent_source,
            GENERATED_AGENT_CONFIG_START,
            GENERATED_AGENT_CONFIG_END,
            agent_config,
        ),
    )
    for path, source, start, end, additions in generated_regions:
        if not _region_contains_only(source, start=start, end=end, statements=additions):
            conflicts.append(
                {
                    "path": path,
                    "operation": "update_region",
                    "reason": (
                        "the starter already has generated or customized tool wiring; "
                        "this command attaches only its first tool"
                    ),
                }
            )

    _plan_tool_files(
        root,
        _first_tool_files(agent_name=agent_name, tool_name=tool_name, effect=effect),
        edits=edits,
        conflicts=conflicts,
        preconditions=preconditions,
    )
    if not conflicts:
        updated_app = _update_region(
            app_source,
            start=GENERATED_IMPORTS_START,
            end=GENERATED_IMPORTS_END,
            additions=app_imports,
        )
        updated_app = _update_region(
            updated_app,
            start=GENERATED_STARTER_TOOLS_START,
            end=GENERATED_STARTER_TOOLS_END,
            additions=app_tool_statements,
        )
        if updated_app != app_source:
            edits.append(
                _edit(
                    registration_relative,
                    "update_region",
                    updated_app,
                    anchor=(f"{GENERATED_IMPORTS_START}; {GENERATED_STARTER_TOOLS_START}"),
                    preimage=app_source,
                )
            )
        else:
            _record_precondition(
                preconditions,
                _file_precondition(root, registration_relative),
            )

        updated_agent = _update_region(
            agent_source,
            start=GENERATED_AGENT_IMPORTS_START,
            end=GENERATED_AGENT_IMPORTS_END,
            additions=agent_imports,
        )
        updated_agent = _update_region(
            updated_agent,
            start=GENERATED_AGENT_CONFIG_START,
            end=GENERATED_AGENT_CONFIG_END,
            additions=agent_config,
        )
        if updated_agent != agent_source:
            edits.append(
                _edit(
                    "agents/agent.py",
                    "update_region",
                    updated_agent,
                    anchor=(f"{GENERATED_AGENT_IMPORTS_START}; {GENERATED_AGENT_CONFIG_START}"),
                    preimage=agent_source,
                )
            )
        else:
            _record_precondition(preconditions, _file_precondition(root, "agents/agent.py"))

    if conflicts:
        status: Literal["ready", "conflict", "manual_action_required", "already_present"] = (
            "conflict"
        )
    elif edits:
        status = "ready"
    else:
        status = "already_present"
    return GeneratorPlan(
        status=status,
        slice_name=agent_name,
        tool_name=tool_name,
        effect=effect,
        edits=tuple(sorted(edits, key=lambda item: item.path)),
        preconditions=tuple(preconditions[path] for path in sorted(preconditions)),
        conflicts=tuple(conflicts),
        verification_commands=_tool_verification_commands(tool_name),
    )


@_guarded_generator_plan
def plan_slice(*, name: str, tool_name: str, effect: str) -> GeneratorPlan:
    name = _identifier(name, "slice name")
    tool_name = _identifier(tool_name, "tool name")
    tool_name_constant = f"{_constant_name(tool_name)}_TOOL_NAME"
    if effect not in {"none", "idempotent", "external"}:
        raise ValueError("effect must be none, idempotent, or external.")
    project = resolve_project(command="cayu generate")
    root = project.root
    registration_relative, registration_path = _registration_target(root)
    if not registration_path.is_file():
        raise ValueError(f"Generated registration target is missing: {registration_relative}.")
    app_content = registration_path.read_bytes()
    app_source = app_content.decode("utf-8")
    app_precondition = GeneratorPrecondition(
        path=registration_relative,
        content_sha256=_sha256(app_content),
    )
    verification = (
        "uv run --no-sync cayu inspect --json",
        "uv run --no-sync cayu check --json",
        f"uv run --no-sync pytest tests/test_{name}.py",
        f"uv run --no-sync cayu eval run evals.{name}:build_eval",
    )
    independent = _slice_files(name=name, tool_name=tool_name, effect=effect)
    tool_imports = [_class_name(tool_name) + "Tool"]
    if effect == "external":
        tool_imports.append(tool_name_constant)
    import_lines = [
        f"from agents.{name} import {_constant_name(name)}_AGENT",
        f"from tools.{tool_name} import {', '.join(tool_imports)}",
    ]
    if effect == "external" and registration_relative != "app.py":
        import_lines.append("from cayu import AlwaysRequireApprovalToolPolicy")
    agent_constant = f"{_constant_name(name)}_AGENT"
    tool_instance = f"{_class_name(tool_name)}Tool()"
    provider_variable = "provider_override" if registration_relative != "app.py" else "provider"
    registration_lines = [
        "app.register_agent(",
        f"    {PROVIDER_OVERRIDE_AGENT_HELPER}({agent_constant}, {provider_variable}),",
        f"    tools=[{tool_instance}],",
    ]
    if effect == "external":
        registration_lines.append(
            f"    tool_policy=AlwaysRequireApprovalToolPolicy(tools=[{tool_name_constant}]),"
        )
    registration_lines.append(")")
    registration = "\n".join(registration_lines)

    conflicts: list[dict[str, str]] = []
    edits: list[GeneratorEdit] = []
    preconditions: dict[str, GeneratorPrecondition] = {}
    proposed_origin = (f"agents.{name}", f"{_constant_name(name)}_AGENT")
    agent_inspection = _normalize_declared_registration_inspection(
        root,
        registration_relative,
        _inspect_registered_agents(
            root,
            app_source,
            source_path=registration_relative,
        ),
    )
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
                "path": registration_relative,
                "operation": "update_region",
                "reason": (
                    f"agent name {name!r} is already registered by {rendered_origins}; "
                    "choose a different slice name or extend the existing agent explicitly"
                ),
            }
        )
    unresolved_conflicts = [
        {
            "path": registration_relative,
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
                "path": registration_relative,
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
                registration_relative,
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
    *,
    source_path: str = "app.py",
) -> _AgentRegistrationInspection:
    """Inspect registered agent identities without importing or executing project code."""

    try:
        app_tree = ast.parse(app_source, filename=source_path)
    except SyntaxError as exc:
        raise ValueError(
            f"Cannot inspect registered agent identities in {source_path}: {exc.msg}."
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
            unresolved.add((source_path, "register_agent"))
            continue
        registered = _unwrap_provider_override_agent(registered)
        if isinstance(registered, ast.Call):
            origin = (source_path, "inline AgentSpec")
            agent_name = _literal_agent_spec_name(registered, app_literals)
            if agent_name is None:
                unresolved.add(origin)
            else:
                registrations.setdefault(agent_name, []).append(origin)
            continue
        if not isinstance(registered, ast.Name):
            unresolved.add((source_path, "register_agent"))
            continue
        if _is_shadowed_in_enclosing_scope(
            node,
            registered.id,
            parents=parents,
        ):
            unresolved.add((source_path, registered.id))
            continue
        origin = agent_imports.get(registered.id)
        if origin is None:
            origin = (source_path, registered.id)
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


def _normalize_declared_registration_inspection(
    root: Path,
    source_path: str,
    inspection: _AgentRegistrationInspection,
) -> _AgentRegistrationInspection:
    """Resolve the maintained coding registration's explicit AgentSpec parameters."""

    if source_path != "agents/registration.py" or _declared_scaffold_preset(root) != "coding":
        return inspection
    parameter_origins = {
        (source_path, "primary_agent"),
        (source_path, "reviewer_agent"),
    }
    if not parameter_origins.issubset(inspection.unresolved_origins):
        return inspection
    if not _coding_registration_helper_is_canonical(root):
        return inspection
    app_precondition = _coding_app_wiring_precondition(root)
    if app_precondition is None:
        return inspection

    origins = {name: list(values) for name, values in inspection.origins_by_name.items()}
    preconditions = {item.path: item for item in inspection.source_preconditions}
    preconditions[app_precondition.path] = app_precondition
    module_snapshots: dict[str, _AgentModuleSnapshot | None] = {}
    for module, symbol in (("agents.agent", "AGENT"), ("agents.reviewer", "REVIEWER")):
        name, precondition = _literal_agent_name(
            root,
            module,
            symbol,
            module_snapshots=module_snapshots,
        )
        if name is None:
            return inspection
        origins.setdefault(name, []).append((module, symbol))
        if precondition is not None:
            preconditions[precondition.path] = precondition
    return _AgentRegistrationInspection(
        origins_by_name={name: tuple(values) for name, values in origins.items()},
        source_preconditions=tuple(preconditions[path] for path in sorted(preconditions)),
        unresolved_origins=inspection.unresolved_origins - parameter_origins,
    )


def _coding_app_wiring_precondition(root: Path) -> GeneratorPrecondition | None:
    relative = "app.py"
    path = _generated_path(root, relative)
    if not path.is_file():
        return None
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
    except (OSError, UnicodeError, SyntaxError):
        return None

    expected_imports = {
        "AGENT": ("agents.agent", "AGENT"),
        "build_coding_app": ("operations.coding", "build_coding_app"),
        PROVIDER_OVERRIDE_AGENT_HELPER: (
            "agents.registration",
            PROVIDER_OVERRIDE_AGENT_HELPER,
        ),
        "REVIEWER": ("agents.reviewer", "REVIEWER"),
    }
    for local_name, (module, symbol) in expected_imports.items():
        if _module_binding_count(tree, local_name) != 1 or not any(
            isinstance(node, ast.ImportFrom)
            and node.module == module
            and any(
                imported.name == symbol and (imported.asname or imported.name) == local_name
                for imported in node.names
            )
            for node in tree.body
        ):
            return None
    factories = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "build_app"
    ]
    if len(factories) != 1:
        return None
    calls = [
        node
        for node in ast.walk(factories[0])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "build_coding_app"
    ]
    if len(calls) != 1:
        return None
    keywords = {keyword.arg: keyword.value for keyword in calls[0].keywords if keyword.arg}
    for field, symbol in (("primary_agent", "AGENT"), ("reviewer_agent", "REVIEWER")):
        value = keywords.get(field)
        if value is None:
            return None
        value = _unwrap_provider_override_agent(value)
        if not isinstance(value, ast.Name) or value.id != symbol:
            return None
    return _file_precondition(root, relative)


def _coding_registration_helper_is_canonical(root: Path) -> bool:
    relative = "agents/registration.py"
    path = _generated_path(root, relative)
    if not path.is_file():
        return False
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
    except (OSError, UnicodeError, SyntaxError):
        return False
    helpers = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == PROVIDER_OVERRIDE_AGENT_HELPER
    ]
    if len(helpers) != 1 or _module_binding_count(tree, PROVIDER_OVERRIDE_AGENT_HELPER) != 1:
        return False
    helper = helpers[0]
    arguments = helper.args
    if (
        helper.decorator_list
        or arguments.posonlyargs
        or [argument.arg for argument in arguments.args] != ["agent", "provider"]
        or arguments.vararg is not None
        or arguments.kwonlyargs
        or arguments.kwarg is not None
        or arguments.defaults
        or arguments.kw_defaults
    ):
        return False
    body = helper.body
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    if len(body) != 2 or not isinstance(body[0], ast.If) or not isinstance(body[1], ast.Return):
        return False
    guard = body[0]
    if (
        not isinstance(guard.test, ast.Compare)
        or not isinstance(guard.test.left, ast.Name)
        or guard.test.left.id != "provider"
        or len(guard.test.ops) != 1
        or not isinstance(guard.test.ops[0], ast.Is)
        or len(guard.test.comparators) != 1
        or not isinstance(guard.test.comparators[0], ast.Constant)
        or guard.test.comparators[0].value is not None
        or len(guard.body) != 1
        or not isinstance(guard.body[0], ast.Return)
        or not isinstance(guard.body[0].value, ast.Name)
        or guard.body[0].value.id != "agent"
        or guard.orelse
    ):
        return False
    returned = body[1].value
    if (
        not isinstance(returned, ast.Call)
        or returned.args
        or len(returned.keywords) != 1
        or returned.keywords[0].arg != "update"
        or not isinstance(returned.func, ast.Attribute)
        or returned.func.attr != "model_copy"
        or not isinstance(returned.func.value, ast.Name)
        or returned.func.value.id != "agent"
    ):
        return False
    update = returned.keywords[0].value
    return (
        isinstance(update, ast.Dict)
        and len(update.keys) == 1
        and isinstance(update.keys[0], ast.Constant)
        and update.keys[0].value == "provider_name"
        and isinstance(update.values[0], ast.Attribute)
        and update.values[0].attr == "name"
        and isinstance(update.values[0].value, ast.Name)
        and update.values[0].value.id == "provider"
    )


def _declared_scaffold_preset(root: Path) -> str | None:
    pyproject = _generated_path(root, "pyproject.toml")
    if not pyproject.is_file():
        return None
    try:
        document = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError):
        return None
    scaffold = document.get("tool", {}).get("cayu", {}).get("scaffold", {})
    preset = scaffold.get("preset")
    return preset if type(preset) is str else None


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


def _unwrap_provider_override_agent(node: ast.expr) -> ast.expr:
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == PROVIDER_OVERRIDE_AGENT_HELPER
        and len(node.args) == 2
        and not node.keywords
    ):
        return node.args[0]
    return node


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
    if origin[0].endswith(".py"):
        return f"{origin[0]}:{origin[1]}"
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


def _region_contains_only(
    source: str,
    *,
    start: str,
    end: str,
    statements: list[str],
) -> bool:
    """Allow an untouched region or the exact idempotent first-tool expansion."""

    bounds = _region_bounds(source, start=start, end=end)
    if bounds is None:
        return False
    body_start, body_end, _ = bounds
    existing, trailing = _parse_region_statements(source[body_start:body_end])
    if trailing.strip():
        return False
    existing_keys = {item.key for item in existing}
    planned_keys = {_statement_key(statement) for statement in statements}
    return existing_keys <= planned_keys


def apply_slice_plan(plan: GeneratorPlan) -> None:
    """Apply a ready slice plan through the durable generator transaction owner."""

    if plan.status != "ready":
        raise GeneratorApplyError(f"Only ready generator plans can be applied, not {plan.status}.")
    project = resolve_project(command="cayu generate")
    try:
        apply_generator_transaction(project.root, _transaction_request(plan))
    except GeneratorTransactionError as exc:
        suffix = f" ({', '.join(exc.paths)})" if exc.paths else ""
        raise GeneratorApplyError(f"{exc}{suffix}") from exc
    except Exception as exc:
        raise GeneratorApplyError(str(exc)) from exc


def apply_service_context_plan(plan: ServiceContextMigrationPlan) -> None:
    """Apply a ready service-context migration through the generator transaction."""

    if plan.status != "ready":
        raise GeneratorApplyError(
            f"Only ready service-context plans can be applied, not {plan.status}."
        )
    apply_slice_plan(
        GeneratorPlan(
            status="ready",
            slice_name="service_context",
            tool_name="service_context",
            effect="none",
            edits=plan.edits,
            preconditions=plan.preconditions,
            conflicts=plan.conflicts,
            verification_commands=plan.verification_commands,
        )
    )


def _transaction_request(plan: GeneratorPlan) -> GeneratorTransactionRequest:
    validate_generator_transaction_collection_bounds(
        edit_count=len(plan.edits),
        precondition_count=len(plan.preconditions),
    )
    remaining = generator_transaction_staged_byte_limit()
    edits: list[GeneratorTransactionEdit] = []
    for edit in plan.edits:
        content = encode_generator_transaction_content(
            edit.content,
            remaining_bytes=remaining,
        )
        remaining -= len(content)
        edits.append(
            GeneratorTransactionEdit(
                path=edit.path,
                operation=edit.operation,
                content=content,
                content_sha256=edit.content_sha256,
                preimage_sha256=edit.preimage_sha256,
            )
        )
    return GeneratorTransactionRequest(
        schema_version=plan.schema_version,
        slice_name=plan.slice_name,
        tool_name=plan.tool_name,
        effect=plan.effect,
        authoring_state=plan.authoring_state.value,
        edits=tuple(edits),
        preconditions=tuple(
            GeneratorTransactionPrecondition(
                path=precondition.path,
                content_sha256=precondition.content_sha256,
            )
            for precondition in plan.preconditions
        ),
        verification_commands=plan.verification_commands,
    )


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

from configuration import configured_model, configured_provider_name

from tools.{tool_name} import {tool_name_constant}


{agent_constant} = AgentSpec(
    name="{name}",
    model=configured_model(),
    provider_name=configured_provider_name(),
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


def _first_tool_files(*, agent_name: str, tool_name: str, effect: str) -> dict[str, str]:
    """Render a tool and its public runtime/eval tracer bullet for an existing agent."""

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
    assert outcome.final_text == "{agent_name} completed sample."'''
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
    tool = f'''from cayu import Tool, ToolContext, ToolEffect, ToolResult, ToolSpec


{tool_name_constant} = "{tool_name}"


class {tool_class}(Tool):
    spec = ToolSpec(
        name={tool_name_constant},
        effect=ToolEffect.{effect_constant},
        description="Process one explicit input for the {agent_name} agent.",
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


def test_{tool_name}_runs_through_public_runtime_seams() -> None:
    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    name={tool_name_constant}, arguments={{"input": "sample"}}
                ),
                ModelStreamEvent.completed({{"finish_reason": "tool_calls"}}),
            ],
            [
                ModelStreamEvent.text_delta("{agent_name} completed sample."),
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
                agent_name="{agent_name}",
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
                ModelStreamEvent.text_delta("{agent_name} completed sample."),
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
            id="{tool_name}-trajectory",
            cases=[
                EvalCase(
                    id="{agent_name}-uses-{tool_name}",
                    request=RunRequest(
                        agent_name="{agent_name}",
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
        f"tools/{tool_name}.py": tool,
        f"tests/test_{tool_name}.py": test,
        f"evals/{tool_name}.py": eval_source,
    }


def _plan_tool_files(
    root: Path,
    files: dict[str, str],
    *,
    edits: list[GeneratorEdit],
    conflicts: list[dict[str, str]],
    preconditions: dict[str, GeneratorPrecondition],
) -> None:
    package_init = "tools/__init__.py"
    package_path = _generated_path(root, package_init)
    if not package_path.exists():
        edits.append(_edit(package_init, "create", ""))
    elif not package_path.is_file():
        conflicts.append(
            {
                "path": package_init,
                "operation": "create",
                "reason": "path exists and is not a regular file",
            }
        )
    else:
        _record_precondition(preconditions, _file_precondition(root, package_init))
    for relative, content in sorted(files.items()):
        path = _generated_path(root, relative)
        if not path.exists():
            edits.append(_edit(relative, "create", content))
        elif not path.is_file() or path.read_text(encoding="utf-8") != content:
            conflicts.append(
                {
                    "path": relative,
                    "operation": "create",
                    "reason": "path exists with user-authored or different content",
                }
            )
        else:
            _record_precondition(preconditions, _file_precondition(root, relative))


def _tool_verification_commands(tool_name: str) -> tuple[str, ...]:
    return (
        "uv run --no-sync cayu inspect --json",
        "uv run --no-sync cayu check --json",
        f"uv run --no-sync pytest tests/test_{tool_name}.py",
        f"uv run --no-sync cayu eval run evals.{tool_name}:build_eval",
    )


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


def _render_service_context_plan(
    plan: ServiceContextMigrationPlan,
    *,
    applied: bool,
) -> str:
    action = "Applied" if applied else "Planned"
    lines: list[str] = [f"{action} service-context: {plan.status}"]
    lines.extend(f"  {edit.operation}: {edit.path}" for edit in plan.edits)
    lines.extend(f"  conflict: {item['path']} — {item['reason']}" for item in plan.conflicts)
    if applied:
        lines.append("Verify:")
        lines.extend(f"  {command}" for command in plan.verification_commands)
    return "\n".join(lines)
