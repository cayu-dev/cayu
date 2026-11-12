from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cayu.cli._check_keys import (
    lent_memory_evidence_key,
    lent_memory_key_diagnostic,
    review_without_keyring_diagnostic,
    uses_tool_argument_review,
)
from cayu.cli._local_alias_keys import keyring_required_diagnostic
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
from cayu.runtime.public_authority import _AliasStoreKeyringRequired
from cayu.runtime.service_manifest import PublicServiceManifest

if TYPE_CHECKING:
    from cayu.applications import CayuApp


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
        report = _with_diagnostics(
            report,
            filter_findings(
                serve_ui_findings(project.root), tags=requested_tags, deploy_only=args.deploy
            ),
        )
        if (
            args.deploy
            and (not requested_tags or "deploy" in requested_tags)
            # Auth resolution can import project code; preserve the source-safety stop.
            and not import_blocking_findings(report.diagnostics)
        ):
            report = _with_diagnostics(report, _cloud_serve_findings(project.root))
    except Exception as exc:
        message = (
            str(exc)
            if isinstance(exc, ProjectError)
            else f"Application factory failed ({type(exc).__name__}): {exc}"
        )
        _render_invocation_error(message, as_json=args.output_format == "json")
        return 2
    return _render_report(args, report)


def serve_ui_findings(root: Path) -> tuple[ProjectDiagnostic, ...]:
    """Check `[tool.cayu.serve] ui`, the built web app `cayu serve` serves at `/`: a
    directory with an `index.html`, and nothing at its top level that Cayu's own routes
    answer. `cayu serve` also refuses one that shadows the project's routes."""

    from cayu._web_app_files import RESERVED_PREFIXES, WebAppError, check_web_app_directory
    from cayu.cli.project import discover_cayu_project_configuration

    discovered = discover_cayu_project_configuration(discovery_keys=("factory",), start=root)
    serve = None if discovered is None else discovered.config.get("serve")
    if not isinstance(serve, dict) or "ui" not in serve:
        return ()
    ui = serve["ui"]

    def finding(code: str, message: str, hint: str) -> ProjectDiagnostic:
        return ProjectDiagnostic(
            code=code,
            severity=DiagnosticSeverity.ERROR,
            subject="cayu serve",
            path="pyproject.toml:[tool.cayu.serve].ui",
            message=message,
            hint=hint,
            tags=("deploy",),
            documentation_anchor="cayu guide references#server",
            verification_command="cayu serve --dev",
        )

    if not isinstance(ui, str) or not ui.strip():
        return (
            finding(
                "SERVE_UI_INVALID",
                "[tool.cayu.serve].ui must be the path of a directory holding a built web app.",
                'Set it to the build\'s output directory, such as ui = "web/static".',
            ),
        )
    # Cayu's API and Developer dashboard, Cayu Cloud's router, and the browser client.
    reserved = (*RESERVED_PREFIXES, "/client.js", "/client.d.ts")
    try:
        check_web_app_directory(root / ui.strip(), reserved)
    except WebAppError as exc:
        missing = exc.reason == "missing"
        return (
            finding(
                "SERVE_UI_MISSING" if missing else "SERVE_UI_SHADOWS_ROUTES",
                str(exc),
                "Build the web app and commit its output, which a deployment serves as it is."
                if missing
                else "Rename that file or folder; Cayu's own routes always answer first.",
            ),
        )
    return ()


def _cloud_serve_findings(root: Path) -> tuple[ProjectDiagnostic, ...]:
    """Check the same local web command and upload inputs as Cloud init/deploy.

    Keep this at the CLI entrance: Cloud's service check itself calls
    ``check_project``, and Doctor/non-deploy checks do not check a Cloud manifest.
    """

    from cayu.cli._cloud_api import CloudApiError
    from cayu.cli._cloud_deploy_check import check_serve_can_start
    from cayu.cli._cloud_project import CloudProjectManifest
    from cayu.cli._serve_readiness import is_cayu_serve_command

    path = root / "cayu-cloud.toml"
    if not path.exists() and not path.is_symlink():
        return ()
    try:
        manifest = CloudProjectManifest.load(path)
    except CloudApiError as exc:
        raise ProjectError(str(exc)) from exc
    if manifest.web is None or not is_cayu_serve_command(manifest.web.command):
        return ()
    findings, reason = check_serve_can_start(root, manifest.web.command, check_auth_source=True)
    if reason is not None and not findings:
        raise ProjectError(reason)
    return findings


def check_project(
    project: CayuProject,
    *,
    tags: frozenset[str] = frozenset(),
    deploy_only: bool = False,
) -> ProjectCheckReport:
    """Check a production project, retaining source-only reports when it cannot load."""

    source_errors = False
    try:
        all_source_diagnostics = check_declared_scaffold_source(
            project.root,
            entry_targets=project.factory_targets,
        )
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
                        blocking_source_diagnostics,
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
            with (
                project_context(project.root),
                lent_memory_evidence_key(project.root) as lent_memory_key,
            ):
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
            key_findings = _key_findings(
                project.root, app, lent_memory_key=lent_memory_key, deploy=deploy_only
            )
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
                entry_targets=project.factory_targets,
                tags=tags,
                deploy_only=deploy_only,
            )
            report = _with_diagnostics(
                report, filter_findings(key_findings, tags=tags, deploy_only=deploy_only)
            )
        finally:
            close_project_control_plane_context(control_plane_context)
    except _AliasStoreKeyringRequired as exc:
        # The local database, not the project's code, stopped the factory.
        return _source_only_report(
            (*source_diagnostics, keyring_required_diagnostic(exc, root=project.root))
        )
    except Exception as exc:
        if source_errors and not isinstance(exc, ProjectError):
            # Source drift often explains a factory failure; keep it visible.
            return _source_only_report(
                (*source_diagnostics, application_load_failed_diagnostic(exc))
            )
        raise

    return report


def _key_findings(
    root: Path, app: CayuApp, *, lent_memory_key: bool, deploy: bool
) -> tuple[ProjectDiagnostic, ...]:
    """Keys this run lacks: a lent memory evidence key, a review without a keyring."""

    findings: list[ProjectDiagnostic] = []
    if lent_memory_key:
        findings.append(lent_memory_key_diagnostic(deploy=deploy))
    if (
        app._public_authority_alias_codec is None
        and not app.human_review_configured
        and (module := uses_tool_argument_review(root)) is not None
    ):
        findings.append(review_without_keyring_diagnostic(module))
    return tuple(findings)


def build_project_check_report(
    project_root: Path,
    manifest: AppManifest,
    *,
    service_manifest: PublicServiceManifest | None,
    project_control_plane: ProjectControlPlaneCheckEvidence,
    entry_targets: tuple[str, ...] = (),
    tags: frozenset[str] = frozenset(),
    deploy_only: bool = False,
) -> ProjectCheckReport:
    """Build the exact structured check report shared by check and doctor.

    *entry_targets* are the factory targets that were loaded, so the source
    findings cover the same modules as the check that ran before the import.
    """

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
        entry_targets=entry_targets,
        tags=tags,
        deploy_only=deploy_only,
    )
    return _with_diagnostics(report, scaffold_diagnostics)


def _with_diagnostics(
    report: ProjectCheckReport, diagnostics: tuple[ProjectDiagnostic, ...]
) -> ProjectCheckReport:
    if not diagnostics:
        return report
    return report.model_copy(
        update={
            "diagnostics": tuple(
                sorted(
                    (*report.diagnostics, *diagnostics),
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
