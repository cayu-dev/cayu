"""Local deploy check run by `cayu cloud deploy` and `cayu cloud init`.

Cayu Cloud starts a public service's web process with `cayu serve`, which refuses to
start a production service while `check_public_service_deployment` reports anything.
This runs the same `cayu check --deploy` report locally, through the check library, so
that refusal shows up before the upload instead of as a web process that exits 1.
"""

from __future__ import annotations

import contextlib
import json
import shlex
import sys
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from cayu._operator_credentials import (
    ENVIRONMENT_OPERATOR_AUTH_TARGET,
    LAST_RELEASE_WITHOUT_ENVIRONMENT_AUTH,
)
from cayu.cli._serve_readiness import (
    GENERATED_AUTH_MODULE,
    declared_runtime_ships_environment_auth,
    environment_auth_upgrade_edits,
    is_cayu_serve_command,
    lock_installs_server_extra,
    serve_auth_state,
    server_extra_edit,
    server_extra_state,
)
from cayu.cli._targets import TargetResolutionError, load_target
from cayu.cli.check import check_project
from cayu.cli.project import (
    ProjectError,
    _project_import_roots,
    discover_cayu_project_configuration,
    project_context,
    resolve_project,
)
from cayu.cli.serve import (
    ServeError,
    parse_serve_arguments,
    require_loopback_dev_host,
    serve_auth_target,
)
from cayu.runtime.checks import (
    PUBLIC_SERVICE_DEPLOYMENT_CODES,
    DiagnosticSeverity,
    ProjectDiagnostic,
)

DEPLOY_CHECK_COMMAND = "cayu check --deploy --fail-on warning --json"
SKIP_DEPLOY_CHECK_FLAG = "--skip-deploy-check"
DEPLOY_CHECK_ENVIRONMENT_HINT = (
    "The check uses this shell's environment. If access or storage is configured through "
    "Cayu Cloud environment variables or secrets, set the same variables here for the "
    f"check, or pass {SKIP_DEPLOY_CHECK_FLAG}."
)

_AUTH_PATH = "pyproject.toml:[tool.cayu.serve].auth"
_WEB_COMMAND_PATH = "cayu-cloud.toml:[web].command"

CloudDeployCheckStatus = Literal["failed", "not_applicable", "passed", "skipped", "unavailable"]


@dataclass(frozen=True)
class CloudDeployCheck:
    """Outcome of the local deploy check for one Cayu Cloud source directory."""

    status: CloudDeployCheckStatus
    blocking: tuple[ProjectDiagnostic, ...] = ()
    reason: str | None = None

    @property
    def codes(self) -> tuple[str, ...]:
        return tuple(item.code for item in self.blocking)

    def public_dict(self) -> dict[str, object]:
        result: dict[str, object] = {"command": DEPLOY_CHECK_COMMAND, "status": self.status}
        if self.blocking:
            result["blocking"] = [
                item.model_dump(
                    mode="json",
                    include={
                        "code",
                        "documentation_anchor",
                        "hint",
                        "message",
                        "path",
                        "severity",
                    },
                )
                for item in self.blocking
            ]
        if self.reason is not None:
            result["reason"] = self.reason
        return result


def run_cloud_deploy_check(
    root: Path,
    *,
    serves_web: bool,
    web_command: str | None = None,
) -> CloudDeployCheck:
    """Check locally that the web process Cayu Cloud starts can start.

    When the web command is `cayu serve`, the project must declare `cayu[server]`
    or an extra that includes it (in `pyproject.toml` and, when present, `uv.lock`).
    The command must not use `--dev`, and, unless a service factory owns access,
    the auth target `cayu serve`
    would load (`--auth`, else `[tool.cayu.serve].auth`) must resolve. These checks
    read files and import only a custom auth target; they don't need the server
    extra installed here.

    For a public service, it also runs `cayu check --deploy`, booting the project in
    production mode with this process's environment, as `cayu check` does. Only the
    findings `cayu serve` refuses to start on are blocking. A project that can't be
    booted here (for example because its dependencies aren't installed in this
    environment) is reported as `unavailable`, not as a failure, since Cloud builds
    the release in its own environment.
    """

    root = root.resolve()
    if not serves_web:
        return CloudDeployCheck(status="not_applicable")
    try:
        configuration = discover_cayu_project_configuration(discovery_keys=("factory",), start=root)
    except ProjectError as exc:
        return CloudDeployCheck(status="unavailable", reason=str(exc))
    # Only the uploaded directory's own project is served; a parent project is not.
    if configuration is None or configuration.root != root:
        return CloudDeployCheck(status="not_applicable")
    serve_blocking: tuple[ProjectDiagnostic, ...] = ()
    serve_reason: str | None = None
    checks_serve = is_cayu_serve_command(web_command)
    if checks_serve:
        assert web_command is not None
        serve_blocking, serve_reason = check_serve_can_start(root, web_command)
    if "service_factory" not in configuration.config:
        if not checks_serve:
            return CloudDeployCheck(status="not_applicable")
        if serve_blocking:
            return CloudDeployCheck(status="failed", blocking=serve_blocking)
        if serve_reason is not None:
            return CloudDeployCheck(status="unavailable", reason=serve_reason)
        return CloudDeployCheck(status="passed")
    service = _run_public_service_check(root)
    blocking = serve_blocking + service.blocking
    if blocking:
        return CloudDeployCheck(status="failed", blocking=blocking)
    return service


def _run_public_service_check(root: Path) -> CloudDeployCheck:
    try:
        project = resolve_project(command="cayu cloud deploy", start=root)
        # Project code may print; keep the CLI's JSON result alone on stdout.
        with contextlib.redirect_stdout(sys.stderr):
            report = check_project(project, deploy_only=True)
    except Exception as exc:
        reason = (
            str(exc)
            if isinstance(exc, ProjectError)
            else f"Application factory failed ({type(exc).__name__}): {exc}"
        )
        return CloudDeployCheck(status="unavailable", reason=reason)
    if report.service_evidence.service_contract != "verified_maintained":
        return CloudDeployCheck(
            status="unavailable",
            reason=(
                "The project check did not evaluate the public service. Findings: "
                + ", ".join(item.code for item in report.diagnostics)
            ),
        )
    blocking = tuple(
        item for item in report.diagnostics if item.code in PUBLIC_SERVICE_DEPLOYMENT_CODES
    )
    return CloudDeployCheck(status="failed" if blocking else "passed", blocking=blocking)


def check_serve_can_start(
    root: Path, web_command: str = "cayu serve"
) -> tuple[tuple[ProjectDiagnostic, ...], str | None]:
    """Find what would stop `web_command`, a `cayu serve` command, starting on Cayu Cloud.

    Returns blocking findings and, when an auth target could not be evaluated
    here, the reason. The command's arguments are parsed by `cayu serve`'s own
    parser, and the effective auth target follows its precedence: `--dev` serves
    without one, and `--auth` overrides `[tool.cayu.serve].auth`. The target may be
    any callable auth dependency. Only the ready-made environment target, when the
    declared cayu release ships it, and the module `cayu cloud init` generates are
    accepted without importing them, because their credentials come from Cayu
    Cloud rather than this shell.
    """

    pyproject = root / "pyproject.toml"
    try:
        text = pyproject.read_text(encoding="utf-8")
        document = tomllib.loads(text)
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
        return (), f"Could not read {pyproject}: {exc}"
    findings: list[ProjectDiagnostic] = []
    extra = server_extra_state(document)
    if extra.status != "present":
        findings.append(
            _serve_diagnostic(
                "SERVE_SERVER_EXTRA_MISSING",
                path="pyproject.toml:[project].dependencies",
                message=(
                    "`cayu serve` needs cayu's server dependencies, but "
                    "[project].dependencies does not declare the `server` extra "
                    "or an extra that includes it, so Cloud would not install them."
                ),
                hint=server_extra_edit(extra)
                + " Then run `uv lock`, or run `cayu cloud init --force`.",
            )
        )
    else:
        project = document.get("project", {})
        lock_path = root / "uv.lock"
        installs = None
        if lock_path.is_file() and isinstance(project.get("name"), str):
            try:
                installs = lock_installs_server_extra(
                    lock_path.read_text(encoding="utf-8"), project["name"]
                )
            except (OSError, UnicodeError):
                installs = None
        if installs is False:
            findings.append(
                _serve_diagnostic(
                    "SERVE_LOCK_MISSING_SERVER_EXTRA",
                    path="uv.lock",
                    message=(
                        "pyproject.toml declares cayu's server dependencies, but uv.lock "
                        "does not, and Cloud installs from uv.lock."
                    ),
                    hint="Run `uv lock` and commit uv.lock.",
                )
            )
    try:
        arguments = parse_serve_arguments(shlex.split(web_command)[2:])
    except (ServeError, ValueError) as exc:
        findings.append(
            _serve_diagnostic(
                "SERVE_COMMAND_INVALID",
                path=_WEB_COMMAND_PATH,
                message=f"`cayu serve` would reject the web command {web_command!r}: {exc}",
                hint="Fix the [web] command in cayu-cloud.toml.",
            )
        )
        return tuple(findings), None
    if arguments.dev:
        findings.append(_dev_finding(arguments.host))
        return tuple(findings), None
    auth = serve_auth_state(document)
    reason: str | None = None
    if auth.status == "service_factory":
        tool = document.get("tool", {})
        serve = tool.get("cayu", {}).get("serve") if isinstance(tool, dict) else None
        configured = isinstance(serve, dict) and "auth" in serve
        if configured or arguments.auth is not None:
            findings.append(
                _serve_diagnostic(
                    "SERVE_AUTH_WITH_SERVICE_FACTORY",
                    path=_AUTH_PATH if configured else _WEB_COMMAND_PATH,
                    message=(
                        "A public service's service_factory owns product and operator access, "
                        "so `cayu serve` refuses [tool.cayu.serve].auth and --auth."
                    ),
                    hint=(
                        "Remove [tool.cayu.serve].auth."
                        if configured
                        else "Remove --auth from the [web] command in cayu-cloud.toml."
                    ),
                )
            )
        return tuple(findings), None
    if auth.status == "invalid":
        # `cayu serve` validates the configured value even when --auth overrides it.
        findings.append(
            _serve_diagnostic(
                "SERVE_AUTH_MISSING",
                path=_AUTH_PATH,
                message="`cayu serve` refuses [tool.cayu.serve].auth unless it is a "
                "non-empty module:attribute string.",
                hint=_auth_setting_hint(document, root),
            )
        )
        return tuple(findings), None
    target = serve_auth_target(arguments, auth.target)
    path = _AUTH_PATH if arguments.auth is None else _WEB_COMMAND_PATH
    if target is None:
        findings.append(
            _serve_diagnostic(
                "SERVE_AUTH_MISSING",
                path=_AUTH_PATH,
                message=(
                    "`cayu serve` refuses to start an unauthenticated server outside --dev, "
                    "and neither [tool.cayu.serve].auth nor --auth names a target."
                ),
                hint=_auth_setting_hint(document, root),
            )
        )
    elif target == ENVIRONMENT_OPERATOR_AUTH_TARGET:
        if not declared_runtime_ships_environment_auth(document, root=root):
            findings.append(_environment_auth_unavailable(document, path=path))
    elif not _is_generated_auth_module(root, target):
        finding, reason = _resolve_auth_target(root, target, path=path)
        if finding is not None:
            findings.append(finding)
    return tuple(findings), reason


def _auth_setting_hint(document: Mapping[str, Any], root: Path) -> str:
    if declared_runtime_ships_environment_auth(document, root=root):
        return (
            f'Set auth = "{ENVIRONMENT_OPERATOR_AUTH_TARGET}" (or your own auth '
            "dependency) under [tool.cayu.serve], or run `cayu cloud init --force`."
        )
    module = GENERATED_AUTH_MODULE
    return (
        f"Run `cayu cloud init --force`, which writes {module.path} and sets auth = "
        f"{json.dumps(module.target)} because the declared cayu requirement allows "
        f"{LAST_RELEASE_WITHOUT_ENVIRONMENT_AUTH} or older, or set [tool.cayu.serve].auth "
        "to your own auth dependency."
    )


def _dev_finding(host: str) -> ProjectDiagnostic:
    try:
        require_loopback_dev_host(host)
    except ServeError as exc:
        message = f"`cayu serve --dev` would refuse to start: {exc}"
    else:
        message = (
            f"`cayu serve --dev` serves an unauthenticated control plane on loopback host "
            f"{host}, which Cayu Cloud cannot route requests to."
        )
    return _serve_diagnostic(
        "SERVE_DEV_MODE",
        path=_WEB_COMMAND_PATH,
        message=message,
        hint=(
            "Remove --dev from the [web] command in cayu-cloud.toml and authenticate "
            "operators with [tool.cayu.serve].auth or --auth."
        ),
    )


def _environment_auth_unavailable(document: Mapping[str, Any], *, path: str) -> ProjectDiagnostic:
    module = GENERATED_AUTH_MODULE
    return _serve_diagnostic(
        "SERVE_AUTH_TARGET_UNRESOLVABLE",
        path=path,
        message=(
            f"`cayu serve` cannot load its auth target {ENVIRONMENT_OPERATOR_AUTH_TARGET}: "
            "the declared cayu requirement allows cayu "
            f"{LAST_RELEASE_WITHOUT_ENVIRONMENT_AUTH} or older, which do not ship that module."
        ),
        hint=" ".join(environment_auth_upgrade_edits(document))
        + " Then run `uv lock`. Or point the auth target at a project module that builds "
        f"BasicAuth, such as the {module.path} `cayu cloud init` writes "
        f"({json.dumps(module.target)}).",
    )


def _is_generated_auth_module(root: Path, target: str) -> bool:
    module = GENERATED_AUTH_MODULE
    if target != module.target:
        return False
    path = root / module.path
    try:
        return (
            not path.is_symlink()
            and not (root / module.path.removesuffix(".py")).exists()
            and path.read_text(encoding="utf-8") == module.content
        )
    except (OSError, UnicodeError):
        return False


def _resolve_auth_target(
    root: Path, target: str, *, path: str
) -> tuple[ProjectDiagnostic | None, str | None]:
    where = "--auth in the [web] command" if path == _WEB_COMMAND_PATH else "[tool.cayu.serve].auth"
    module_name = target.partition(":")[0]
    local = module_name.partition(".")[0] in _project_import_roots(root)
    try:
        with project_context(root), contextlib.redirect_stdout(sys.stderr):
            auth = load_target(target, label="Serve authentication target", normalize_errors=True)
    except TargetResolutionError as exc:
        if "module was not found" in str(exc) and not local:
            # It may come from a dependency Cloud installs but this environment lacks.
            return None, f"{exc} It may be installed on Cayu Cloud but not here."
        return (
            _serve_diagnostic(
                "SERVE_AUTH_TARGET_UNRESOLVABLE",
                path=path,
                message=f"`cayu serve` cannot load its auth target {target}: {exc}",
                hint=f"Point {where} at a module:attribute in the project.",
            ),
            None,
        )
    except (Exception, SystemExit) as exc:
        return None, (
            f"The auth target {target} could not be loaded here ({type(exc).__name__}: {exc}). "
            "It may need credentials or packages that only Cayu Cloud has."
        )
    if not callable(auth):
        return (
            _serve_diagnostic(
                "SERVE_AUTH_TARGET_UNRESOLVABLE",
                path=path,
                message=f"`cayu serve` auth target {target} is not callable.",
                hint="Point it at an auth dependency that takes a request and returns AuthContext.",
            ),
            None,
        )
    return None, None


def _serve_diagnostic(code: str, *, path: str, message: str, hint: str) -> ProjectDiagnostic:
    return ProjectDiagnostic(
        code=code,
        severity=DiagnosticSeverity.ERROR,
        subject="cayu serve",
        path=path,
        message=message,
        hint=hint,
        tags=("deploy",),
        documentation_anchor="cayu guide references#server",
        verification_command="cayu serve --host 0.0.0.0 --port 8000",
    )


def deploy_check_refusal(check: CloudDeployCheck) -> str:
    """The refusal `cayu cloud deploy` reports for a failed check."""

    serve = [item for item in check.blocking if item.code.startswith("SERVE_")]
    service = [item for item in check.blocking if not item.code.startswith("SERVE_")]
    reasons: list[str] = []
    if serve:
        reasons.append(
            "its `cayu serve` web process could not start: "
            + " ".join(f"{item.message} {item.hint}" for item in serve)
        )
    if service:
        reasons.append(
            "`cayu serve` refuses the public service because the deploy check reports "
            + ", ".join(item.code for item in service)
            + f"; fix the findings and rerun `{DEPLOY_CHECK_COMMAND}`"
        )
    return (
        "This Agent would not start on Cayu Cloud: "
        + "; and ".join(reasons)
        + f". Nothing was uploaded. Pass {SKIP_DEPLOY_CHECK_FLAG} to upload anyway."
    )


def init_deploy_check_notice(check: CloudDeployCheck) -> str | None:
    """What `cayu cloud init` tells the author about the deploy check, if anything."""

    if check.status == "failed":
        serve = [item for item in check.blocking if item.code.startswith("SERVE_")]
        service = [item for item in check.blocking if not item.code.startswith("SERVE_")]
        parts: list[str] = []
        if serve:
            parts.append(
                "Its `cayu serve` web process can't start yet: "
                + " ".join(f"{item.message} {item.hint}" for item in serve)
            )
        if service:
            placeholder = any(
                item.parameters.get("configured") == "placeholder" for item in service
            )
            reason = (
                " Its product or operator access is still a placeholder." if placeholder else ""
            )
            parts.append(
                "This public service won't start on Cayu Cloud yet: the deploy check reports "
                f"{', '.join(item.code for item in service)}, so `cayu serve` refuses it and "
                f"`cayu cloud deploy` won't upload it.{reason} Configure it, then run "
                f"`{DEPLOY_CHECK_COMMAND}` until it passes. {DEPLOY_CHECK_ENVIRONMENT_HINT}"
            )
        return " ".join(parts)
    if check.status == "unavailable":
        return (
            "This Agent's web process could not be fully checked here"
            + (f": {check.reason}" if check.reason else ".")
            + f" Run `{DEPLOY_CHECK_COMMAND}` in the project environment before deploying; "
            "`cayu serve` refuses to start a service it reports problems for."
        )
    return None
