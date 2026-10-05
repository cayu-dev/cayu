from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from cayu.cli._output import add_output_options, output_destination
from cayu.cli.project import (
    CayuProject,
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
    application_load_failed_diagnostic,
    application_not_imported_diagnostic,
    check_declared_scaffold,
    check_declared_scaffold_source,
    filter_findings,
    import_blocking_findings,
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
from cayu.runtime.manifest import AppManifest
from cayu.runtime.service_manifest import PublicServiceManifest


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
        report = check_project(project, tags=requested_tags, deploy_only=args.deploy)
    except Exception as exc:
        message = (
            str(exc)
            if isinstance(exc, ProjectError)
            else f"Application factory failed ({type(exc).__name__}): {exc}"
        )
        _render_invocation_error(message, as_json=args.output_format == "json")
        return 2
    return _render_report(args, report)


def check_project(
    project: CayuProject,
    *,
    tags: frozenset[str] = frozenset(),
    deploy_only: bool = False,
) -> ProjectCheckReport:
    """Check a production project, retaining source-only reports when it cannot load."""

    source_errors = False
    try:
        all_source_diagnostics = check_declared_scaffold_source(project.root)
        source_diagnostics = filter_findings(
            all_source_diagnostics,
            tags=tags,
            deploy_only=deploy_only,
        )
        # Never import code that failed the import-safety check, whatever the
        # requested tags. Blocking findings are always reported, even outside the
        # requested tags: they stopped every other check, so a tagged run must
        # not pass having checked nothing. Other source errors still let the
        # app load and be checked.
        blocking_source_diagnostics = import_blocking_findings(all_source_diagnostics)
        if blocking_source_diagnostics:
            outside_selection = tuple(
                item for item in blocking_source_diagnostics if item not in source_diagnostics
            )
            return _source_only_report(
                (
                    *source_diagnostics,
                    *outside_selection,
                    application_not_imported_diagnostic(
                        len(blocking_source_diagnostics),
                        outside_selection=len(outside_selection),
                    ),
                )
            )
        # Only reported errors may stand in for a factory failure below.
        source_errors = any(
            item.severity is DiagnosticSeverity.ERROR for item in source_diagnostics
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
            report = build_project_check_report(
                project.root,
                manifest,
                service_manifest=None if service is None else service.manifest,
                project_control_plane=check_evidence,
                tags=tags,
                deploy_only=deploy_only,
            )
        finally:
            close_project_control_plane_context(control_plane_context)
    except Exception as exc:
        if source_errors and not isinstance(exc, ProjectError):
            # Source drift often explains a factory failure; keep it visible.
            return _source_only_report(
                (*source_diagnostics, application_load_failed_diagnostic(exc))
            )
        raise

    return report


def build_project_check_report(
    project_root: Path,
    manifest: AppManifest,
    *,
    service_manifest: PublicServiceManifest | None,
    project_control_plane: ProjectControlPlaneCheckEvidence,
    tags: frozenset[str] = frozenset(),
    deploy_only: bool = False,
) -> ProjectCheckReport:
    """Build the exact structured check report shared by check and doctor."""

    report = check_manifest(
        manifest,
        service_manifest=service_manifest,
        project_control_plane=project_control_plane,
        tags=tags,
        deploy_only=deploy_only,
    )
    scaffold_diagnostics = check_declared_scaffold(
        project_root,
        manifest,
        tags=tags,
        deploy_only=deploy_only,
    )
    if not scaffold_diagnostics:
        return report
    return report.model_copy(
        update={
            "diagnostics": tuple(
                sorted(
                    (*report.diagnostics, *scaffold_diagnostics),
                    key=lambda item: (item.severity.value, item.code, item.path),
                )
            )
        }
    )


def _source_only_report(
    diagnostics: tuple[ProjectDiagnostic, ...],
) -> ProjectCheckReport:
    """Return stable findings without claiming that a project manifest was loaded."""

    return ProjectCheckReport(
        manifest_fingerprint="unavailable",
        diagnostics=tuple(
            sorted(diagnostics, key=lambda item: (item.severity.value, item.code, item.path))
        ),
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
