from __future__ import annotations

import argparse
import hashlib
import json
import keyword
import re
import shutil
import sys
import tempfile
from contextlib import suppress
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

_IDENTIFIER_RE = re.compile(r"[a-z][a-z0-9_]*")
GENERATOR_PLAN_SCHEMA_VERSION = "1"


class GeneratorEdit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    operation: Literal["create", "update_region"]
    content: str
    content_sha256: str
    preimage_sha256: str | None = None
    anchor: str | None = None


class GeneratorPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = GENERATOR_PLAN_SCHEMA_VERSION
    status: Literal["ready", "conflict", "manual_action_required", "already_present"]
    slice_name: str
    tool_name: str
    effect: str
    edits: tuple[GeneratorEdit, ...]
    conflicts: tuple[dict[str, str], ...] = ()
    verification_commands: tuple[str, ...]


class GeneratorApplyError(RuntimeError):
    """The planned generator transaction could not be applied safely."""


class _GeneratedPathError(ValueError):
    pass


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
        help="Declared ToolEffect for the generated tool.",
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
    app_source = app_path.read_text(encoding="utf-8")
    verification = (
        "cayu inspect --json",
        "cayu check --json",
        f"pytest tests/test_{name}.py",
        f"cayu eval run evals.{name}:build_eval",
    )
    independent = _slice_files(name=name, tool_name=tool_name, effect=effect)

    conflicts: list[dict[str, str]] = []
    edits: list[GeneratorEdit] = []
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
        elif not path.is_file() or path.read_text(encoding="utf-8") != content:
            conflicts.append(
                {
                    "path": relative,
                    "operation": "create",
                    "reason": "path exists with user-authored or different content",
                }
            )

    tool_imports = [_class_name(tool_name) + "Tool"]
    if effect == "external":
        tool_imports.append(tool_name_constant)
    import_lines = [
        f"from agents.{name} import {_constant_name(name)}_AGENT",
        f"from tools.{tool_name} import {', '.join(tool_imports)}",
    ]
    if effect == "external":
        import_lines.append("from cayu import AlwaysRequireApprovalToolPolicy")
    registration = (
        f"app.register_agent({_constant_name(name)}_AGENT, tools=[{_class_name(tool_name)}Tool()]"
    )
    if effect == "external":
        registration += (
            f", tool_policy=AlwaysRequireApprovalToolPolicy(tools=[{tool_name_constant}])"
        )
    registration += ")"

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
    if conflicts:
        status: Literal["ready", "conflict", "manual_action_required", "already_present"] = (
            "conflict"
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
        conflicts=tuple(conflicts),
        verification_commands=verification,
    )


def apply_slice_plan(plan: GeneratorPlan) -> None:
    """Apply a ready slice plan as an all-or-nothing filesystem transaction."""

    if plan.status != "ready":
        raise GeneratorApplyError(f"Only ready generator plans can be applied, not {plan.status}.")
    project = resolve_project(command="cayu generate")
    root = project.root.resolve()
    targets: list[tuple[GeneratorEdit, Path]] = []
    seen: set[Path] = set()
    try:
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
    else:
        test_assertions = f'''    assert outcome.ok
    assert outcome.final_text == "{name} completed sample."'''
        eval_assertions = f"""                        SessionCompleted(),
                        ToolCalled({tool_name_constant}),
                        FinalOutputContains("sample"),"""
    agent = f'''from cayu import AgentSpec

from tools.{tool_name} import {tool_name_constant}


{agent_constant} = AgentSpec(
    name="{name}",
    model="gpt-5.4-mini",
    system_prompt=f"Use {{{tool_name_constant}}} when it directly answers the user's request.",
    workflow_tool_names=({tool_name_constant},),
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
    EventType,
    InMemorySessionStore,
    InMemoryTaskStore,
    Message,
    ModelStreamEvent,
    RunRequest,
    ScriptedModelProvider,
    SessionStatus,
    run_to_completion,
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
    EvalCase,
    EvalPlan,
    EvalSuite,
    EventOccurred,
    EventType,
    FinalOutputContains,
    InMemorySessionStore,
    InMemoryTaskStore,
    Message,
    ModelStreamEvent,
    RunRequest,
    ScriptedModelProvider,
    SessionCompleted,
    SessionInterrupted,
    ToolCalled,
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


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _update_region(source: str, *, start: str, end: str, additions: list[str]) -> str:
    start_index = source.index(start)
    line_start = source.rfind("\n", 0, start_index) + 1
    indent = source[line_start:start_index]
    body_start = source.index("\n", start_index) + 1
    end_index = source.index(end, body_start)
    end_line_start = source.rfind("\n", 0, end_index) + 1
    existing = [
        line.strip() for line in source[body_start:end_line_start].splitlines() if line.strip()
    ]
    lines = sorted(set(existing).union(additions))
    body = "".join(f"{indent}{line}\n" for line in lines)
    return source[:body_start] + body + source[end_line_start:]


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
