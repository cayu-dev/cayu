"""Convert an existing scaffolded project to another preset.

``cayu generate preset`` renders the project's declared plan and the requested
plan with the same templates ``cayu new`` uses. A file that still matches what
the declared plan generates is replaced, created, or removed to match the
requested plan; a file the user changed is never overwritten, and its path is
reported as a conflict so nothing is written.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import secrets
import shutil
import sys
import tempfile
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict

from cayu.cli import scaffold as scaffold_cli
from cayu.cli._generator_transaction import (
    GeneratorTransactionError,
    generator_planning_guard,
    recover_generator_transaction,
)
from cayu.cli._output import add_output_options
from cayu.cli.generate import (
    GENERATOR_PLAN_SCHEMA_VERSION,
    GeneratorApplyError,
    GeneratorEdit,
    GeneratorPlan,
    GeneratorPrecondition,
    _edit,
    _generated_path,
    _guarded_generator_plan,
    _sha256,
    apply_slice_plan,
)
from cayu.cli.project import ProjectError, resolve_project
from cayu.cli.scaffold import (
    _NAME_RE,
    _private_files,
    _ScaffoldCommandError,
    project_files,
)
from cayu.cli.scaffold_check import _scaffold_contract
from cayu.cli.scaffold_convention import _creation_command, local_database_path
from cayu.cli.scaffold_plan import (
    ADAPTERS,
    PRESETS,
    ApplicationPlan,
    ScaffoldPlanError,
    normalize_application_plan,
    preset_spec,
)
from cayu.workspaces.local import LocalWorkspace


class PresetPlanSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    preset: str
    provider: str
    execution: str
    capabilities: tuple[str, ...]


class PresetConversionPlan(BaseModel):
    """One reviewable conversion of a scaffolded project to another preset."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = GENERATOR_PLAN_SCHEMA_VERSION
    status: Literal["ready", "conflict", "manual_action_required", "already_present"]
    migration: Literal["preset"] = "preset"
    source: PresetPlanSummary | None = None
    target: PresetPlanSummary | None = None
    edits: tuple[GeneratorEdit, ...] = ()
    removals: tuple[GeneratorPrecondition, ...] = ()
    directories: tuple[str, ...] = ()
    private_files: tuple[str, ...] = ()
    unchanged: tuple[str, ...] = ()
    conflicts: tuple[dict[str, str], ...] = ()
    notes: tuple[str, ...] = ()
    verification_commands: tuple[str, ...] = ()


def add_preset_parser(generators: argparse._SubParsersAction) -> None:
    parser = generators.add_parser(
        "preset",
        help="Convert this scaffolded project to another preset, such as coding with Docker.",
        description=(
            "Convert this scaffolded project to another preset with the same templates "
            "`cayu new` uses. Files you changed are never overwritten: they are reported "
            "as conflicts and nothing is written. Use `--dry-run` to review the plan."
        ),
    )
    parser.add_argument(
        "preset",
        choices=tuple(spec.name for spec in PRESETS),
        help="Target preset.",
    )
    parser.add_argument(
        "--execution",
        choices=tuple(spec.name for spec in ADAPTERS if spec.kind == "execution"),
        help="Target execution adapter (default: the project's declared execution).",
    )
    parser.add_argument(
        "--provider",
        choices=tuple(spec.name for spec in ADAPTERS if spec.kind == "provider"),
        help="Target provider adapter (default: the project's declared provider).",
    )
    parser.add_argument(
        "--with",
        dest="with_capabilities",
        action="append",
        default=[],
        metavar="CAPABILITY",
        help="Enable a maintained optional capability, as with `cayu new`.",
    )
    parser.add_argument(
        "--without",
        dest="without_capabilities",
        action="append",
        default=[],
        metavar="CAPABILITY",
        help="Turn off a default capability of the target preset, as with `cayu new`.",
    )
    parser.add_argument(
        "--agent-name",
        help="Registered first-agent name (default: the name in agents/agent.py).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Plan without writes.")
    add_output_options(parser)


def run_generate_preset(args: argparse.Namespace) -> int:
    try:
        project = resolve_project(command="cayu generate preset")
        recover_generator_transaction(project.root, dry_run=args.dry_run)
        plan = plan_preset_conversion(
            preset=args.preset,
            execution=args.execution,
            provider=args.provider,
            with_capabilities=tuple(args.with_capabilities),
            without_capabilities=tuple(args.without_capabilities),
            agent_name=args.agent_name,
        )
    except (GeneratorTransactionError, ProjectError, ValueError, OSError) as exc:
        return _failure("GENERATOR_PLAN_FAILED", str(exc), args.output_format)
    should_apply = not args.dry_run and plan.status == "ready"
    if should_apply:
        try:
            apply_preset_conversion(plan)
        except (GeneratorApplyError, ProjectError, OSError) as exc:
            return _failure("GENERATOR_APPLY_FAILED", str(exc), args.output_format)
    if args.output_format == "json":
        print(plan.model_dump_json(indent=2))
    else:
        print(render_preset_conversion(plan, applied=should_apply))
    return 0 if plan.status in {"ready", "already_present"} else 1


def _failure(code: str, message: str, output_format: str) -> int:
    if output_format == "json":
        print(
            json.dumps(
                {
                    "schema_version": GENERATOR_PLAN_SCHEMA_VERSION,
                    "error": {"code": code, "message": message},
                },
                sort_keys=True,
            )
        )
    else:
        print(f"error: {message}", file=sys.stderr)
    return 2


def _manual(reason: str, *, path: str = "pyproject.toml") -> PresetConversionPlan:
    return PresetConversionPlan(
        status="manual_action_required",
        conflicts=({"path": path, "reason": reason},),
    )


@_guarded_generator_plan
def plan_preset_conversion(
    *,
    preset: str,
    execution: str | None = None,
    provider: str | None = None,
    with_capabilities: tuple[str, ...] = (),
    without_capabilities: tuple[str, ...] = (),
    agent_name: str | None = None,
) -> PresetConversionPlan:
    """Plan the conversion of the current project without writing anything."""

    root = resolve_project(command="cayu generate preset").root
    try:
        document = tomllib.loads(_generated_path(root, "pyproject.toml").read_text("utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
        return _manual(f"pyproject.toml could not be read: {exc}")
    contract = _scaffold_contract(document)
    if contract is None:
        return _manual(
            "there is no [tool.cayu.scaffold] contract; only projects created by "
            "`cayu new` can be converted"
        )
    if contract.get("convention") != 1:
        return _manual(f"scaffold convention {contract.get('convention')!r} is not supported")
    project_table = document.get("project")
    name = project_table.get("name") if isinstance(project_table, Mapping) else None
    if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
        return _manual(f"[project].name {name!r} is not a project name `cayu new` accepts")
    # The source is rebuilt with the name the project has; an override renames
    # only the target.
    declared_agent_name = _declared_agent_name(root) or name
    if agent_name is None:
        agent_name = declared_agent_name
    if not _NAME_RE.fullmatch(agent_name):
        raise ValueError(f"invalid agent name {agent_name!r}")
    try:
        source = _declared_plan(contract, name=name, agent_name=declared_agent_name)
    except ScaffoldPlanError as exc:
        return _manual(f"the declared scaffold plan is invalid: {exc}")
    try:
        target = normalize_application_plan(
            name=name,
            agent_name=agent_name,
            preset=preset,
            provider=provider or source.provider,
            execution=execution or source.execution,
            with_capabilities=with_capabilities,
            without_capabilities=without_capabilities,
        )
    except ScaffoldPlanError as exc:
        raise ValueError(str(exc)) from None

    current_files = project_files(name, application_plan=source)
    target_files = project_files(name, application_plan=target)
    edits: list[GeneratorEdit] = []
    removals: list[GeneratorPrecondition] = []
    unchanged: list[str] = []
    conflicts: list[dict[str, str]] = []
    for relative in sorted(set(current_files) | set(target_files)):
        generated = current_files.get(relative)
        wanted = target_files.get(relative)
        try:
            existing = _read_regular_file(root, relative)
        except ValueError as exc:
            conflicts.append({"path": relative, "reason": str(exc)})
            continue
        generated_bytes = None if generated is None else generated.encode("utf-8")
        if wanted is None:
            if existing is None:
                continue
            if existing == generated_bytes:
                removals.append(
                    GeneratorPrecondition(path=relative, content_sha256=_sha256(existing))
                )
            else:
                conflicts.append(
                    {
                        "path": relative,
                        "reason": (
                            f"changed since it was generated, and the {target.preset} preset "
                            "doesn't generate it; move or restore it"
                        ),
                    }
                )
            continue
        wanted_bytes = wanted.encode("utf-8")
        if existing == wanted_bytes:
            unchanged.append(relative)
        elif existing is None:
            edits.append(_edit(relative, "create", wanted))
        elif existing == generated_bytes:
            # existing equals generated, so the preimage digest is the same.
            edits.append(_edit(relative, "update_region", wanted, preimage=generated))
        else:
            conflicts.append(
                {
                    "path": relative,
                    "reason": (
                        "changed since it was generated"
                        if generated is not None
                        else "already exists and isn't the generated file"
                    )
                    + "; restore it, or move your change aside and merge it back after",
                }
            )

    directories = tuple(
        directory
        for directory in (
            ("data", "data/artifacts") if "artifacts" in target.capabilities else ("data",)
        )
        if not (root / directory).is_dir()
    )
    private_files = tuple(path for path in _private_files(target) if not (root / path).exists())
    if conflicts:
        status: Literal["ready", "conflict", "already_present"] = "conflict"
    elif edits or removals or directories or private_files:
        status = "ready"
    else:
        status = "already_present"
    return PresetConversionPlan(
        status=status,
        source=_summary(source),
        target=_summary(target),
        edits=tuple(edits),
        removals=tuple(removals),
        directories=directories,
        private_files=private_files,
        unchanged=tuple(unchanged),
        conflicts=tuple(conflicts),
        notes=_notes(root, source, target, conflicted=bool(conflicts)),
        verification_commands=target.verification_commands(),
    )


def apply_preset_conversion(plan: PresetConversionPlan) -> None:
    """Apply a ready conversion; edits go through the generator transaction."""

    if plan.status != "ready" or plan.target is None:
        raise GeneratorApplyError(f"Only ready preset plans can be applied, not {plan.status}.")
    root = resolve_project(command="cayu generate preset").root
    if plan.target.preset == "coding":
        _preflight_coding()
    try:
        with generator_planning_guard(root):
            _require_planned_files(root, plan)
            # Remove before the edits commit. Until then the declared plan still
            # generates these files, so a run interrupted anywhere plans the same
            # edits again and finishes; removing them after the commit would leave
            # them behind, outside both plans, if the process died in between.
            for removal in plan.removals:
                path = _generated_path(root, removal.path)
                if os.path.lexists(path):
                    path.unlink()
                # Drop directories the removal emptied, such as memory/.
                for parent in Path(removal.path).parents:
                    if parent == Path("."):
                        break
                    directory = _generated_path(root, parent.as_posix())
                    if directory.is_dir() and not any(directory.iterdir()):
                        directory.rmdir()
    except (GeneratorTransactionError, ValueError) as exc:
        raise GeneratorApplyError(str(exc)) from exc
    if plan.edits:
        apply_slice_plan(
            GeneratorPlan(
                status="ready",
                slice_name=f"preset_{plan.target.preset}",
                tool_name="preset",
                effect="none",
                edits=plan.edits,
                verification_commands=plan.verification_commands,
            )
        )
    try:
        with generator_planning_guard(root):
            for directory in plan.directories:
                _generated_path(root, directory).mkdir(parents=True, exist_ok=True)
            for relative in plan.private_files:
                _write_private_key(_generated_path(root, relative))
    except (GeneratorTransactionError, ValueError) as exc:
        raise GeneratorApplyError(str(exc)) from exc


def _require_planned_files(root: Path, plan: PresetConversionPlan) -> None:
    """Refuse before any change when a planned file changed since planning."""

    expected = [(edit.path, edit.preimage_sha256) for edit in plan.edits]
    expected += [(removal.path, removal.content_sha256) for removal in plan.removals]
    for relative, digest in expected:
        existing = _read_regular_file(root, relative)
        current = None if existing is None else _sha256(existing)
        if current != digest and not (current is None and relative in _removed(plan)):
            raise GeneratorApplyError(
                f"{relative} changed after the plan was made; nothing was changed, plan again."
            )


def _removed(plan: PresetConversionPlan) -> frozenset[str]:
    return frozenset(removal.path for removal in plan.removals)


def render_preset_conversion(plan: PresetConversionPlan, *, applied: bool) -> str:
    action = "Applied" if applied else "Planned"
    target = plan.target
    heading = (
        "preset" if target is None else f"preset {target.preset} (execution {target.execution})"
    )
    lines = [f"{action} {heading}: {plan.status}"]
    for label, summary in (("from", plan.source), ("to", plan.target)):
        if summary is not None:
            lines.append(
                f"  {label + ':':<5} preset={summary.preset} provider={summary.provider} "
                f"execution={summary.execution}"
            )
    if target is not None:
        lines.append("  capabilities: " + (", ".join(target.capabilities) or "none"))
    operations = {"create": "create", "update_region": "update"}
    lines.extend(f"  {operations[edit.operation]}: {edit.path}" for edit in plan.edits)
    lines.extend(f"  remove: {removal.path}" for removal in plan.removals)
    lines.extend(f"  create directory: {directory}" for directory in plan.directories)
    lines.extend(f"  create private: {path}" for path in plan.private_files)
    if plan.unchanged:
        lines.append(f"  unchanged: {len(plan.unchanged)} files")
    lines.extend(f"  conflict: {item['path']} — {item['reason']}" for item in plan.conflicts)
    lines.extend(f"  note: {note}" for note in plan.notes)
    if applied:
        lines.append("Verify:")
        lines.extend(f"  {command}" for command in plan.verification_commands)
    return "\n".join(lines)


def _declared_plan(
    contract: Mapping[str, object], *, name: str, agent_name: str
) -> ApplicationPlan:
    for field in ("preset", "provider", "execution"):
        if type(contract.get(field)) is not str:
            raise ScaffoldPlanError("invalid_" + field, f"{field} must be a string")
    recorded = contract.get("capabilities")
    if not isinstance(recorded, list) or any(type(item) is not str for item in recorded):
        raise ScaffoldPlanError("invalid_capabilities", "capabilities must be a list of strings")
    preset = cast("str", contract["preset"])
    defaults = set(preset_spec(preset).default_capabilities)
    selected = set(cast("list[str]", recorded))
    plan = normalize_application_plan(
        name=name,
        agent_name=agent_name,
        preset=preset,
        provider=cast("str", contract["provider"]),
        execution=cast("str", contract["execution"]),
        with_capabilities=tuple(sorted(selected - defaults)),
        without_capabilities=tuple(sorted(defaults - selected)),
    )
    if list(plan.capabilities) != recorded:
        raise ScaffoldPlanError(
            "capabilities_not_normalized",
            "capabilities must match the normalized selected plan",
        )
    return plan


def _declared_agent_name(root: Path) -> str | None:
    """Read the literal ``AGENT = AgentSpec(name=...)`` the scaffold generates."""

    try:
        tree = ast.parse(_generated_path(root, "agents/agent.py").read_text("utf-8"))
    except (OSError, UnicodeError, SyntaxError, ValueError):
        return None
    for node in tree.body:
        if not (
            isinstance(node, ast.Assign)
            and [target.id for target in node.targets if isinstance(target, ast.Name)] == ["AGENT"]
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "AgentSpec"
        ):
            continue
        for keyword in node.value.keywords:
            if (
                keyword.arg == "name"
                and isinstance(keyword.value, ast.Constant)
                and isinstance(keyword.value.value, str)
            ):
                return keyword.value.value
    return None


def _read_regular_file(root: Path, relative: str) -> bytes | None:
    path = _generated_path(root, relative)
    if not os.path.lexists(path):
        return None
    if path.is_symlink() or not path.is_file():
        raise ValueError("is not a regular file")
    return path.read_bytes()


def _summary(plan: ApplicationPlan) -> PresetPlanSummary:
    return PresetPlanSummary(
        preset=plan.preset,
        provider=plan.provider,
        execution=plan.execution,
        capabilities=plan.capabilities,
    )


def _notes(
    root: Path, source: ApplicationPlan, target: ApplicationPlan, *, conflicted: bool
) -> tuple[str, ...]:
    notes: list[str] = []
    old_database, new_database = local_database_path(source), local_database_path(target)
    if old_database != new_database:
        notes.append(
            f"The local SQLite database moves from {old_database} to {new_database}. "
            f"Sessions, tasks and knowledge in {old_database} are left in place and the "
            "converted app won't see them; copy the file to the new path before the first "
            "run to keep them. Deployments that set CAYU_DATABASE_URL are unaffected."
        )
    if conflicted:
        reference = _creation_command(target, name=target.name).removesuffix(" --json")
        notes.append(
            "Nothing will be written while files conflict. To see the generated version "
            f"of a file, run `{reference} --agent-name {target.agent_name} "
            '--dir "$(mktemp -d)"` and compare.'
        )
    if target.preset == "coding":
        if not (root / ".git").exists():
            notes.append(
                "This project isn't a Git repository. The coding agent works on a "
                "committed Git checkout: run `git init -b main` and commit before a live run."
            )
        else:
            notes.append(
                "Commit the converted files before a live run; the coding agent starts "
                "from a clean Git checkout."
            )
        if target.execution == "docker":
            notes.append(
                "Before proof: uv run --no-sync python build_coding_image.py --resolve-pins, "
                "then review docker-coding-build.json."
            )
    return tuple(notes)


def _preflight_coding() -> None:
    """Check the Git and ripgrep the coding preset needs, as `cayu new` does."""

    parent = Path(tempfile.mkdtemp(prefix="cayu-generate-preset-"))
    try:
        LocalWorkspace.require_path_operations_supported()
        scaffold_cli._preflight_coding_commands(parent=parent)
    except (RuntimeError, _ScaffoldCommandError, OSError) as exc:
        raise GeneratorApplyError(f"coding preset {exc}") from None
    finally:
        shutil.rmtree(parent, ignore_errors=True)


def _write_private_key(path: Path) -> None:
    if os.path.lexists(path):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(secrets.token_urlsafe(48) + "\n")
