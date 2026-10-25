from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any

from cayu.cli._output import add_output_options, output_destination
from cayu.cli.project import ProjectError, build_project_app, project_context, resolve_project
from cayu.cli.scaffold_check import check_scaffold_capabilities
from cayu.runtime.manifest import APP_MANIFEST_SCHEMA_VERSION, AppManifest


def add_inspect_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "inspect",
        help="Describe a booted Cayu project without running it.",
        description=(
            "Describe a booted Cayu project without running it. This proves the "
            "structural manifest only; use `cayu check` for diagnostics."
        ),
    )
    parser.add_argument(
        "target",
        nargs="?",
        help="Override project discovery with a module:factory target.",
    )
    add_output_options(parser)
    subjects = parser.add_mutually_exclusive_group()
    subjects.add_argument("--agent", help="Return one agent from the manifest.")
    subjects.add_argument("--tool", help="Return agents registering this tool.")
    subjects.add_argument("--environment", help="Return one environment from the manifest.")
    subjects.add_argument(
        "--retry-key",
        help=(
            "Read one task retry key's budget state from the project's task store: "
            "attempts, the budget, and whether it gave up."
        ),
    )


def run_inspect(args: argparse.Namespace) -> int:
    try:
        with output_destination(args.output):
            return _run_inspect(args)
    except OSError as exc:
        print(f"error: could not write output: {exc}", file=sys.stderr)
        return 2


def _run_inspect(args: argparse.Namespace) -> int:
    try:
        project = resolve_project(args.target, command="cayu inspect")
        with project_context(project.root):
            app = build_project_app(project.target, command="Inspect")
            if args.retry_key is not None:
                return _inspect_retry_key(app, args.retry_key, args.output_format)
            manifest = app.describe(project_root=project.root)
    except Exception as exc:
        if args.output_format == "json":
            print(
                json.dumps(
                    {
                        "schema_version": APP_MANIFEST_SCHEMA_VERSION,
                        "error": {
                            "code": "PROJECT_BOOT_FAILED",
                            "message": _project_error_message(exc),
                        },
                    },
                    sort_keys=True,
                )
            )
        else:
            print(f"error: {_project_error_message(exc)}", file=sys.stderr)
        return 2

    filtered, error = _filter_manifest(manifest, args)
    if error is not None:
        if args.output_format == "json":
            print(
                json.dumps(
                    {"schema_version": APP_MANIFEST_SCHEMA_VERSION, "error": error},
                    sort_keys=True,
                )
            )
        else:
            print(f"error: {error['message']}", file=sys.stderr)
        return 1

    diagnostics = check_scaffold_capabilities(project.root, manifest)
    if diagnostics:
        if args.output_format == "json":
            print(
                json.dumps(
                    {
                        "schema_version": APP_MANIFEST_SCHEMA_VERSION,
                        "error": {
                            "code": diagnostics[0].code,
                            "message": "Declared capabilities disagree with the constructed application.",
                        },
                        "diagnostics": [item.model_dump(mode="json") for item in diagnostics],
                        "manifest": filtered.model_dump(mode="json"),
                    },
                    sort_keys=True,
                )
            )
        else:
            for item in diagnostics:
                print(f"{item.code}: {item.path}: {item.message}", file=sys.stderr)
        return 1

    if args.output_format == "json":
        print(filtered.model_dump_json(indent=2))
    else:
        print(_render_human(filtered))
    return 0


def _inspect_retry_key(app: Any, retry_key: str, output_format: str) -> int:
    async def load() -> Any:
        try:
            if app.task_store is None or not app.task_store.supports_task_retry_keys:
                raise ProjectError("The project's task store does not support task retry keys.")
            return await app.task_store.load_task_retry_key(retry_key)
        finally:
            await app.aclose()
            # The CLI process owns the stores its project factory opened.
            close = getattr(app.task_store, "close", None)
            if close is not None:
                await close()

    state = asyncio.run(load())
    if state is None:
        error = {
            "code": "SUBJECT_NOT_FOUND",
            "message": f"Retry key not found: {retry_key}.",
            "path": f"retry_keys.{retry_key}",
        }
        if output_format == "json":
            print(json.dumps({"schema_version": APP_MANIFEST_SCHEMA_VERSION, "error": error}))
        else:
            print(f"error: {error['message']}", file=sys.stderr)
        return 1
    if output_format == "json":
        print(
            json.dumps(
                {"schema_version": APP_MANIFEST_SCHEMA_VERSION, "retry_key": state.inspection()},
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    budget = state.budget
    cap = "no attempt cap" if budget.max_attempts is None else f"of {budget.max_attempts}"
    window = (
        "no time cap"
        if budget.max_elapsed_seconds is None
        else f"within {budget.max_elapsed_seconds:g}s of the first"
    )
    lines = [
        f"Retry key {state.retry_key}: {state.status.value}",
        f"Attempts: {state.attempts} {cap}, {window}",
        f"First attempt: {state.first_attempt_at.isoformat()}",
        f"Latest task: {state.latest_task_id} at {state.latest_attempt_at.isoformat()}",
    ]
    if state.given_up_at is not None and state.given_up_reason is not None:
        lines.append(
            f"Gave up: {state.given_up_at.isoformat()} ({state.given_up_reason.value}); "
            f"refused task {state.refused_task_id}"
        )
    print("\n".join(lines))
    return 0


def _render_human(manifest: AppManifest) -> str:
    agents = ", ".join(item.name for item in manifest.agents) or "none"
    providers = ", ".join(item.name for item in manifest.providers) or "none"
    environments = (
        ", ".join(
            f"{item.name} (deferred)" if item.deferred else item.name
            for item in manifest.environments
        )
        or "none"
    )
    return "\n".join(
        (
            f"Cayu application {manifest.fingerprint[:12]}",
            f"Agents: {agents}",
            f"Providers: {providers}",
            f"Environments: {environments}",
            "Verification: structural inspection only; no live capability was verified.",
        )
    )


def _filter_manifest(
    manifest: AppManifest,
    args: argparse.Namespace,
) -> tuple[AppManifest, dict[str, str] | None]:
    if args.agent is not None:
        agents = tuple(item for item in manifest.agents if item.name == args.agent)
        if not agents:
            return manifest, {
                "code": "SUBJECT_NOT_FOUND",
                "message": f"Agent not found: {args.agent}.",
                "path": f"agents.{args.agent}",
            }
        return manifest.model_copy(
            update={"agents": agents, "providers": (), "environments": ()}
        ), None
    if args.tool is not None:
        agents = tuple(
            agent.model_copy(
                update={"tools": tuple(tool for tool in agent.tools if tool.name == args.tool)}
            )
            for agent in manifest.agents
            if any(tool.name == args.tool for tool in agent.tools)
        )
        if not agents:
            return manifest, {
                "code": "SUBJECT_NOT_FOUND",
                "message": f"Tool not found: {args.tool}.",
                "path": f"tools.{args.tool}",
            }
        return manifest.model_copy(
            update={"agents": agents, "providers": (), "environments": ()}
        ), None
    if args.environment is not None:
        environments = tuple(
            item for item in manifest.environments if item.name == args.environment
        )
        if not environments:
            return manifest, {
                "code": "SUBJECT_NOT_FOUND",
                "message": f"Environment not found: {args.environment}.",
                "path": f"environments.{args.environment}",
            }
        return manifest.model_copy(
            update={"agents": (), "providers": (), "environments": environments}
        ), None
    return manifest, None


def _project_error_message(exc: Exception) -> str:
    if isinstance(exc, ProjectError):
        return str(exc)
    return f"Application factory failed ({type(exc).__name__}): {exc}"
