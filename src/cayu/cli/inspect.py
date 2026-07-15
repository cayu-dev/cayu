from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from cayu.cli.project import ProjectError, build_project_app, project_context, resolve_project
from cayu.runtime.manifest import APP_MANIFEST_SCHEMA_VERSION, AppManifest


def add_inspect_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "inspect",
        help="Describe a booted Cayu project without running it.",
    )
    parser.add_argument(
        "target",
        nargs="?",
        help="Override project discovery with a module:factory target.",
    )
    parser.add_argument("--json", action="store_true", help="Emit the stable JSON manifest.")
    subjects = parser.add_mutually_exclusive_group()
    subjects.add_argument("--agent", help="Return one agent from the manifest.")
    subjects.add_argument("--tool", help="Return agents registering this tool.")
    subjects.add_argument("--environment", help="Return one environment from the manifest.")


def run_inspect(args: argparse.Namespace) -> int:
    try:
        project = resolve_project(args.target, command="cayu inspect")
        with project_context(project.root):
            app = build_project_app(project.target, command="Inspect")
            manifest = app.describe(project_root=project.root)
    except Exception as exc:
        if args.json:
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
        if args.json:
            print(
                json.dumps(
                    {"schema_version": APP_MANIFEST_SCHEMA_VERSION, "error": error},
                    sort_keys=True,
                )
            )
        else:
            print(f"error: {error['message']}", file=sys.stderr)
        return 1

    if args.json:
        print(filtered.model_dump_json(indent=2))
    else:
        print(_render_human(filtered))
    return 0


def _render_human(manifest: AppManifest) -> str:
    agents = ", ".join(item.name for item in manifest.agents) or "none"
    providers = ", ".join(item.name for item in manifest.providers) or "none"
    environments = ", ".join(item.name for item in manifest.environments) or "none"
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
