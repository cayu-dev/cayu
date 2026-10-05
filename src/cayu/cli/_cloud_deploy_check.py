"""Local deploy check run by `cayu cloud deploy` and `cayu cloud init`.

Cayu Cloud starts a public service's web process with `cayu serve`, which refuses to
start a production service while `check_public_service_deployment` reports anything.
This runs the same `cayu check --deploy` report locally, through the check library, so
that refusal shows up before the upload instead of as a web process that exits 1.
"""

from __future__ import annotations

import contextlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from cayu.cli.check import check_project
from cayu.cli.project import ProjectError, discover_cayu_project_configuration, resolve_project
from cayu.runtime.checks import PUBLIC_SERVICE_DEPLOYMENT_CODES, ProjectDiagnostic

DEPLOY_CHECK_COMMAND = "cayu check --deploy --fail-on warning --json"
SKIP_DEPLOY_CHECK_FLAG = "--skip-deploy-check"
DEPLOY_CHECK_ENVIRONMENT_HINT = (
    "The check uses this shell's environment. If access or storage is configured through "
    "Cayu Cloud environment variables or secrets, set the same variables here for the "
    f"check, or pass {SKIP_DEPLOY_CHECK_FLAG}."
)

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


def run_cloud_deploy_check(root: Path, *, serves_web: bool) -> CloudDeployCheck:
    """Run `cayu check --deploy` for a Cayu public service that Cloud serves on the web.

    The check boots the project in production mode with this process's environment, as
    `cayu check` does. Only the findings `cayu serve` refuses to start on are blocking.
    A project that can't be booted here (for example because its dependencies aren't
    installed in this environment) is reported as `unavailable`, not as a failure, since
    Cloud builds the release in its own environment.
    """

    root = root.resolve()
    if not serves_web:
        return CloudDeployCheck(status="not_applicable")
    try:
        configuration = discover_cayu_project_configuration(discovery_keys=("factory",), start=root)
    except ProjectError as exc:
        return CloudDeployCheck(status="unavailable", reason=str(exc))
    # Only the uploaded directory's own project is served; a parent project is not.
    if (
        configuration is None
        or configuration.root != root
        or "service_factory" not in configuration.config
    ):
        return CloudDeployCheck(status="not_applicable")
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


def deploy_check_refusal(check: CloudDeployCheck) -> str:
    """The refusal `cayu cloud deploy` reports for a failed check."""

    return (
        "This public service would not start on Cayu Cloud: `cayu serve` refuses it "
        f"because the deploy check reports {', '.join(check.codes)}. Nothing was uploaded. "
        f"Fix the findings and rerun `{DEPLOY_CHECK_COMMAND}`, or pass "
        f"{SKIP_DEPLOY_CHECK_FLAG} to upload anyway."
    )


def init_deploy_check_notice(check: CloudDeployCheck) -> str | None:
    """What `cayu cloud init` tells the author about the deploy check, if anything."""

    if check.status == "failed":
        placeholder = any(
            item.parameters.get("configured") == "placeholder" for item in check.blocking
        )
        reason = " Its product or operator access is still a placeholder." if placeholder else ""
        return (
            "This public service won't start on Cayu Cloud yet: the deploy check reports "
            f"{', '.join(check.codes)}, so `cayu serve` refuses it and `cayu cloud deploy` "
            f"won't upload it.{reason} Configure it, then run `{DEPLOY_CHECK_COMMAND}` until "
            f"it passes. {DEPLOY_CHECK_ENVIRONMENT_HINT}"
        )
    if check.status == "unavailable":
        return (
            "This public service could not be checked here. Run "
            f"`{DEPLOY_CHECK_COMMAND}` in the project environment before deploying; "
            "`cayu serve` refuses to start a service it reports problems for."
        )
    return None
