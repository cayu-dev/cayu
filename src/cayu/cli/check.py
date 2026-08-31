from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from cayu.cli._output import add_output_options, output_destination
from cayu.cli.project import (
    ProjectError,
    build_project_app,
    build_project_service,
    project_context,
    resolve_project,
)
from cayu.cli.project_control_plane import (
    build_project_control_plane_context,
    close_project_control_plane_context,
)
from cayu.cli.scaffold_check import (
    check_declared_scaffold,
    check_declared_scaffold_source,
)
from cayu.runtime.checks import (
    AVAILABLE_CHECK_TAGS,
    DiagnosticSeverity,
    ProjectCheckReport,
    ProjectControlPlaneCheckEvidence,
    ProjectDiagnostic,
    ServiceCheckEvidence,
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
        source_diagnostics = check_declared_scaffold_source(
            project.root,
            tags=requested_tags,
            deploy_only=args.deploy,
        )
        blocking_source_diagnostics = tuple(
            item for item in source_diagnostics if item.severity is DiagnosticSeverity.ERROR
        )
        if blocking_source_diagnostics:
            return _render_report(
                args,
                _source_only_report(blocking_source_diagnostics),
            )
        control_plane_context = build_project_control_plane_context(
            project.root,
            mode="production",
        )
        try:
            with project_context(project.root):
                service = (
                    None
                    if project.service_target is None
                    else build_project_service(
                        project.service_target,
                        mode="production",
                        command="Check",
                        project_context=control_plane_context,
                    )
                )
                app = (
                    build_project_app(project.target, command="Check")
                    if service is None
                    else service.cayu_app
                )
                manifest = app.describe(project_root=project.root)
            check_evidence = ProjectControlPlaneCheckEvidence(
                project_identity_configured=(control_plane_context.project_identity_configured),
                eval_store_configured=control_plane_context.eval_store_configured,
                service_context=(
                    "not_applicable"
                    if service is None
                    else (
                        "attached"
                        if service.project_control_plane_context_attached
                        else "migration_required"
                    )
                ),
            )
            report = check_manifest(
                manifest,
                service_manifest=None if service is None else service.manifest,
                project_control_plane=check_evidence,
                tags=requested_tags,
                deploy_only=args.deploy,
            )
            scaffold_diagnostics = check_declared_scaffold(
                project.root,
                manifest,
                tags=requested_tags,
                deploy_only=args.deploy,
            )
            if scaffold_diagnostics:
                report = report.model_copy(
                    update={
                        "diagnostics": tuple(
                            sorted(
                                (*report.diagnostics, *scaffold_diagnostics),
                                key=lambda item: (
                                    item.severity.value,
                                    item.code,
                                    item.path,
                                ),
                            )
                        )
                    }
                )
        finally:
            close_project_control_plane_context(control_plane_context)
    except Exception as exc:
        message = (
            str(exc)
            if isinstance(exc, ProjectError)
            else f"Application factory failed ({type(exc).__name__}): {exc}"
        )
        _render_invocation_error(message, as_json=args.output_format == "json")
        return 2

    return _render_report(args, report)


def _source_only_report(
    diagnostics: tuple[ProjectDiagnostic, ...],
) -> ProjectCheckReport:
    """Return stable findings without claiming that a project manifest was loaded."""

    return ProjectCheckReport(
        manifest_fingerprint="unavailable",
        diagnostics=tuple(sorted(diagnostics, key=lambda item: (item.code, item.path))),
        service_evidence=ServiceCheckEvidence(
            control_plane_access="not_evaluated",
            service_contract="not_declared",
            application_security="not_evaluated",
            configuration="not_applicable",
            host_owned_behavior="unverified_outside_contract",
            security_verification_command="pytest -q tests/test_public_service_security.py",
        ),
    )


def _render_report(args: argparse.Namespace, report: ProjectCheckReport) -> int:
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
