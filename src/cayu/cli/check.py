from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from cayu.cli._output import add_output_options, output_destination
from cayu.cli.project import ProjectError, build_project_app, project_context, resolve_project
from cayu.runtime.checks import (
    AVAILABLE_CHECK_TAGS,
    DiagnosticSeverity,
    ProjectCheckReport,
    check_manifest,
    severity_at_least,
)


def add_check_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "check",
        help="Validate a booted Cayu project with actionable diagnostics.",
        description=(
            "Validate a booted Cayu project with actionable diagnostics. "
            "Run `cayu guide diagnostics` to interpret stable finding codes."
        ),
    )
    parser.add_argument(
        "target",
        nargs="?",
        help="Override project discovery with a module:factory target.",
    )
    add_output_options(parser)
    parser.add_argument(
        "--tag",
        action="append",
        default=[],
        help="Run checks carrying this tag (repeatable).",
    )
    parser.add_argument(
        "--deploy",
        action="store_true",
        help="Run only checks that gate deployment.",
    )
    parser.add_argument(
        "--fail-on",
        choices=tuple(item.value for item in DiagnosticSeverity),
        default=DiagnosticSeverity.ERROR.value,
        help="Lowest severity that exits 1 (default: error).",
    )


def run_check(args: argparse.Namespace) -> int:
    try:
        with output_destination(args.output):
            return _run_check(args)
    except OSError as exc:
        print(f"error: could not write output: {exc}", file=sys.stderr)
        return 2


def _run_check(args: argparse.Namespace) -> int:
    requested_tags = frozenset(args.tag)
    unknown = requested_tags - AVAILABLE_CHECK_TAGS
    if unknown:
        message = f"Unknown check tags: {', '.join(sorted(unknown))}."
        _render_invocation_error(message, as_json=args.output_format == "json")
        return 2
    try:
        project = resolve_project(args.target, command="cayu check")
        with project_context(project.root):
            app = build_project_app(project.target, command="Check")
            manifest = app.describe(project_root=project.root)
        report = check_manifest(
            manifest,
            tags=requested_tags,
            deploy_only=args.deploy,
        )
    except Exception as exc:
        message = (
            str(exc)
            if isinstance(exc, ProjectError)
            else f"Application factory failed ({type(exc).__name__}): {exc}"
        )
        _render_invocation_error(message, as_json=args.output_format == "json")
        return 2

    if args.output_format == "json":
        print(report.model_dump_json(indent=2))
    else:
        print(_render_human(report))
    threshold = DiagnosticSeverity(args.fail_on)
    return (
        1 if any(severity_at_least(item.severity, threshold) for item in report.diagnostics) else 0
    )


def _render_invocation_error(message: str, *, as_json: bool) -> None:
    if as_json:
        print(
            json.dumps(
                {
                    "schema_version": "1",
                    "error": {"code": "PROJECT_CHECK_FAILED", "message": message},
                },
                sort_keys=True,
            )
        )
    else:
        print(f"error: {message}", file=sys.stderr)


def _render_human(report: ProjectCheckReport) -> str:
    if not report.diagnostics:
        return f"OK: no qualifying findings ({report.manifest_fingerprint[:12]})."
    lines = []
    for item in report.diagnostics:
        lines.append(f"{item.severity.value.upper()} {item.code} {item.path}: {item.message}")
        if item.hint:
            lines.append(f"  Fix: {item.hint}")
        if item.documentation_anchor:
            lines.append(f"  Docs: {item.documentation_anchor}")
        lines.append(f"  Verify: {item.verification_command}")
    return "\n".join(lines)
