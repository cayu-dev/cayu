"""First-party customer CLI for Cayu Cloud."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import secrets
import shlex
import sys
import time
import webbrowser
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Never

from cayu.cli._cloud_api import CloudApiClient, CloudApiError, archived_agent_error
from cayu.cli._cloud_auth import (
    CloudAuthCredentials,
    CloudAuthError,
    CloudAuthStore,
    WorkOSDeviceAuthClient,
    fresh_cloud_credentials,
)
from cayu.cli._cloud_diagnostics import CloudDeploymentFailure as _CloudDeploymentFailure
from cayu.cli._cloud_diagnostics import parse_build_failure, safe_text
from cayu.cli._cloud_evidence import EvidenceRecorder
from cayu.cli._cloud_private_state import write_private_json as _write_private_json
from cayu.cli._cloud_project import (
    CloudSourceInputsError,
    ResolvedCloudProject,
    initialize_project,
    is_application_slug,
    resolve_project,
)

_DEPLOYMENT_FAILURES = {"cancelled", "destroyed", "failed"}
_DEPLOYMENT_IN_PROGRESS = {
    "accepted",
    "image_built",
    "image_scanned",
    "policy_compiled",
    "sandbox_template_ready",
    "source_resolved",
}
_DEPLOYMENT_READY = {"smoke_tested", "promoted"}
_DEPLOYMENT_STATUSES = _DEPLOYMENT_FAILURES | _DEPLOYMENT_IN_PROGRESS | _DEPLOYMENT_READY
_SERVICE_FAILURES = {"degraded", "failed", "stopped"}
_SERVICE_IN_PROGRESS = {"deleting", "deploying", "sleeping", "starting", "stopping"}
_SERVICE_READY = {"running"}
# `archived` is readable but never waited on: archive retired the service for good.
_SERVICE_STATUSES = _SERVICE_FAILURES | _SERVICE_IN_PROGRESS | _SERVICE_READY | {"archived"}
# A service or archive wait tolerates this many consecutive throttled, unavailable or 5xx reads.
_SERVICE_POLL_TRANSIENT_LIMIT = 5
_TRANSIENT_HTTP_STATUSES = {429, 500, 502, 503, 504}
_CLOUD_DEPLOYMENT_ID = re.compile(r"dep_[a-z0-9]{1,64}")
_PRODUCTION_API_URL = "https://cloud.cayu.dev"
# Cloud's bounds on the breaking Cayu storage revisions one Release acknowledges.
_MAX_STORAGE_REVISION = 1_000_000
_MAX_STORAGE_ACKNOWLEDGEMENTS = 32
_STORAGE_ACKNOWLEDGEMENT_REQUIRED = "storage_breaking_acknowledgement_required"
# Publication failures that a retry of the same Release can get past.
_RETRYABLE_STORAGE_PUBLICATION_FAILURES = frozenset(
    {"storage_migration_state_unknown", "storage_writers_not_stopped"}
)
# Cloud's documented breaking-storage publication failures, reported as `error.code`.
_STORAGE_PUBLICATION_FAILURES = _RETRYABLE_STORAGE_PUBLICATION_FAILURES | {
    _STORAGE_ACKNOWLEDGEMENT_REQUIRED,
    "storage_newer_than_release",
}
_ACKNOWLEDGE_BREAKING_LIST = re.compile(r'"acknowledge_breaking":\s*\[([0-9, ]{1,400})\]')
_ACKNOWLEDGE_BREAKING_FLAG = re.compile(r"--acknowledge-breaking ([0-9]{1,7})\b")
_LEGACY_RETRY_REJECTION = "Only paused or failed deployments can be retried."
# Cloud's owner choice for the serving release's unfinished sessions.
_SESSION_POLICIES = ("wait", "block", "proceed")
_DEFAULT_SESSION_WAIT_SECONDS = 900
_MIN_SESSION_WAIT_SECONDS = 60
_MAX_SESSION_WAIT_SECONDS = 3600
_MAX_SESSION_ACKNOWLEDGEMENTS = 200
_MAX_SESSION_ID_LENGTH = 256
_ACKNOWLEDGE_ALL_SESSIONS = "*"
# Publication refusals for unfinished sessions; a retry with a new choice gets past them.
_SESSION_PUBLICATION_FAILURES = frozenset({"unfinished_sessions", "sessions_unreadable"})
# Session checks after which Cloud publishes, whatever choice is stored later.
_PASSED_SESSION_CHECKS = frozenset({"clear", "acknowledged", "unsupported", "not_running"})
# A waiting preflight's `waiting_for` (Cloud names sessions, session_read or agent_wake).
_SESSION_WAITING_FOR = re.compile(r"[a-z][a-z0-9_]{0,63}")


class CloudCommandError(RuntimeError):
    """Stable customer-facing Cloud command failure."""

    def __init__(self, category: str, message: str) -> None:
        super().__init__(message)
        self.category = category


class _CloudDeployCheckError(CloudCommandError):
    """The local deploy check found problems that stop the public service starting."""

    def __init__(self, message: str, *, details: dict[str, object]) -> None:
        super().__init__("deploy_check_failed", message)
        self.details = details.copy()


class _CloudServiceHealthError(CloudApiError):
    """Safe structured Agent service health failure."""

    def __init__(
        self,
        category: str,
        message: str,
        *,
        issues: Sequence[dict[str, Any]],
    ) -> None:
        super().__init__(category, message)
        self.issues = [dict(issue) for issue in issues]


class _CloudServicePublicationError(CloudApiError):
    """Cayu Cloud promoted the release but could not publish it to the service."""

    def __init__(
        self,
        publication_error: dict[str, Any],
        *,
        application_id: str | None = None,
        deployment_id: str | None = None,
        recovery_arguments: Sequence[str] = (),
        release: dict[str, Any] | None = None,
    ) -> None:
        app = None if application_id is None else _cloud_application_id(application_id)
        deployment = None if deployment_id is None else _cloud_deployment_id(deployment_id)
        self.details: dict[str, object] = {}
        if app is not None:
            self.details["application"] = app
        if deployment is not None:
            self.details["deployment_id"] = deployment
        parts = [
            publication_error[key] for key in ("message", "detail") if key in publication_error
        ]
        code = publication_error.get("code")
        revisions = (
            _requested_storage_acknowledgement(publication_error)
            if code == _STORAGE_ACKNOWLEDGEMENT_REQUIRED
            else ()
        )
        flags = [
            item for revision in revisions for item in ("--acknowledge-breaking", str(revision))
        ]
        sessions: list[str] = (
            _session_retry_acknowledgements(publication_error, release) or []
            if code in _SESSION_PUBLICATION_FAILURES
            else []
        )
        retry_command = None
        wait_command = None
        if app is not None and deployment is not None:
            command = ["cayu", "cloud", *recovery_arguments, "deployment"]
            suffix = [deployment, "--application", app]
            commands = {
                action: shlex.join([*command, action, *suffix]) for action in ("status", "timeline")
            }
            if revisions:
                retry_command = shlex.join([*command, "retry", *suffix, *flags])
            elif code in _RETRYABLE_STORAGE_PUBLICATION_FAILURES:
                retry_command = shlex.join([*command, "retry", *suffix])
            elif sessions:
                retry_command = shlex.join(
                    [
                        *command,
                        "retry",
                        *suffix,
                        *(
                            item
                            for session in sessions
                            for item in ("--acknowledge-session", session)
                        ),
                    ]
                )
            if code in _SESSION_PUBLICATION_FAILURES:
                # A retry replaces the Release's choice, so keep a longer wait it had.
                waited = _stored_session_wait_seconds(release)
                wait_command = shlex.join(
                    [
                        *command,
                        "retry",
                        *suffix,
                        "--session-policy",
                        "wait",
                        *(
                            ["--session-wait-seconds", str(waited)]
                            if waited > _DEFAULT_SESSION_WAIT_SECONDS
                            else []
                        ),
                    ]
                )
                commands["retry_wait"] = wait_command
            if retry_command is not None:
                commands["retry"] = retry_command
            self.details["commands"] = commands
        if code in _SESSION_PUBLICATION_FAILURES and wait_command is not None:
            unfinished = code == "unfinished_sessions"
            wait_step = (
                "wait for them again" if unfinished else "wait for Cloud to read them"
            ) + f" with `{wait_command}`"
            if retry_command is not None:
                # Cloud's hint names the portal first for older CLIs; give this CLI's steps.
                self.details["acknowledge_sessions"] = sessions
                everyone = sessions == [_ACKNOWLEDGE_ALL_SESSIONS]
                parts.append(
                    (
                        "Answer or finish the sessions and retry, or "
                        if unfinished
                        else "Retry once the Agent answers, or "
                    )
                    + f"accept that {'every unfinished session' if everyone else 'they'} may "
                    + f"not resume on this release with `{retry_command}`, or {wait_step}."
                )
            else:
                # Without a list of sessions a retry can pass with (Cloud's is missing,
                # invalid or truncated), keep Cloud's hint (the portal's Acknowledge and
                # retry) and add the wait alternative.
                parts.extend(
                    [
                        *([publication_error["hint"]] if "hint" in publication_error else []),
                        f"Or {wait_step}.",
                    ]
                )
        elif revisions:
            # Cloud's hint names its operator tooling; give the customer CLI's own steps.
            self.details["acknowledge_breaking"] = list(revisions)
            parts.append(
                "A breaking migration stops the previous release while the database migrates, "
                "until this release starts, and releases built with the older Cayu can't be "
                "rolled back to afterwards. To proceed, "
                + (f"run `{retry_command}`, or " if retry_command is not None else "")
                + f"run `cayu cloud deploy` again with `{' '.join(flags)}`."
            )
        elif "hint" in publication_error:
            parts.append(publication_error["hint"])
        super().__init__(
            "service_publication_failed",
            " ".join(parts),
            code=code
            if code in _STORAGE_PUBLICATION_FAILURES or code in _SESSION_PUBLICATION_FAILURES
            else None,
        )
        self.publication_error = publication_error.copy()


class _CloudRetrySubmissionError(CloudApiError):
    def __init__(self, cause: CloudApiError, *, deployment_id: str, retry_key: str):
        super().__init__(cause.category, str(cause), status_code=cause.status_code)
        self.details = {"deployment_id": deployment_id, "retry_idempotency_key": retry_key}


class _CloudPromotionConflictError(CloudApiError):
    """Cloud refused the CLI's promote and has not promoted the release since."""

    def __init__(
        self,
        cause: CloudApiError,
        *,
        application_id: str,
        deployment_id: str,
        recovery_arguments: Sequence[str],
        status: str,
    ) -> None:
        reason = f" Cayu Cloud said: {cause.detail}" if cause.detail is not None else ""
        super().__init__(
            "deployment_promotion_conflict",
            f"Cayu Cloud refused to promote this release and has not promoted it.{reason}",
            status_code=cause.status_code,
            detail=cause.detail,
        )
        self.details: dict[str, object] = {"status": status}
        app = _cloud_application_id(application_id)
        deployment = _cloud_deployment_id(deployment_id)
        if app is not None:
            self.details["application"] = app
        if deployment is not None:
            self.details["deployment_id"] = deployment
        if app is not None and deployment is not None:
            command = ["cayu", "cloud", *recovery_arguments, "deployment"]
            suffix = [deployment, "--application", app]
            self.details["commands"] = {
                action: shlex.join([*command, action, *suffix])
                for action in ("status", "timeline", "promote")
            }


class _CloudDeploymentFailureError(CloudApiError):
    """Safe structured Cayu Cloud deployment failure."""

    def __init__(
        self,
        category: str,
        message: str,
        *,
        failure: _CloudDeploymentFailure,
        details: dict[str, object],
    ) -> None:
        super().__init__(category, message)
        self.failure = failure.copy()
        self.details = details.copy()


class _CloudDeploymentDiagnosticUnavailableError(CloudApiError):
    def __init__(self, details: dict[str, object]) -> None:
        super().__init__(
            "deployment_failed", f"Deployment reached terminal status: {details['status']}"
        )
        self.details = details


class _CloudDeploymentStillRunningError(CloudApiError):
    """A local wait ended while Cayu Cloud retained the deployment operation."""

    def __init__(
        self,
        *,
        application_id: str,
        deployment_id: str,
        recovery_arguments: Sequence[str] = (),
        status: str,
        last_issue: str | None = None,
    ) -> None:
        message = "Cayu Cloud is still processing this deployment."
        if last_issue is not None:
            message = f"{message} Last report: {last_issue}"
        super().__init__("deployment_still_running", message)
        self.application_id = _cloud_application_id(application_id)
        self.deployment_id = _cloud_deployment_id(deployment_id)
        self.recovery_arguments = tuple(recovery_arguments)
        self.status = status if status in _DEPLOYMENT_IN_PROGRESS else "processing"
        self.last_issue = last_issue

    def public_details(self) -> dict[str, object]:
        details: dict[str, object] = {"status": self.status}
        if self.last_issue is not None:
            details["last_issue"] = self.last_issue
        if self.application_id is not None:
            details["application"] = self.application_id
        if self.deployment_id is not None:
            details["deployment_id"] = self.deployment_id
        if self.application_id is not None and self.deployment_id is not None:
            command = ["cayu", "cloud", *self.recovery_arguments, "deployment"]
            suffix = [self.deployment_id, "--application", self.application_id]
            details["commands"] = {
                action: shlex.join([*command, action, *suffix])
                for action in ("status", "timeline", "wait")
            }
        return details


class _CloudReleaseNotSelectedError(CloudApiError):
    """Unchanged source resolved to a promoted Release the Agent no longer runs."""

    def __init__(
        self,
        *,
        application_id: str,
        deployment_id: str,
        current_deployment_id: object,
        recovery_arguments: Sequence[str] = (),
    ) -> None:
        app = _cloud_application_id(application_id)
        deployment = _cloud_deployment_id(deployment_id)
        current = (
            _cloud_deployment_id(current_deployment_id)
            if isinstance(current_deployment_id, str)
            else None
        )
        selected = (
            f"the Agent's selected Release is {current}"
            if current is not None
            else "the Agent has no selected Release"
        )
        message = (
            f"This source is already Release {deployment or 'in Cayu Cloud'}, so Cayu Cloud "
            f"reused it instead of building a new one, but {selected}. Select the reused "
            "Release with `cayu cloud rollback`, or change `version` in cayu-cloud.toml to "
            "build a new Release."
        )
        super().__init__("release_not_selected", message)
        self.details: dict[str, object] = {"current_deployment_id": current}
        if app is not None:
            self.details["application"] = app
        if deployment is not None:
            self.details["deployment_id"] = deployment
        if app is not None and deployment is not None:
            command = ["cayu", "cloud", *recovery_arguments]
            self.details["commands"] = {
                "rollback": shlex.join(
                    [*command, "rollback", deployment, "--application", app, "--wait"]
                ),
                "status": shlex.join([*command, "service", "status", "--application", app]),
            }


class _CloudReleaseSupersededError(CloudApiError):
    """Cloud selected another Release while this one waited for unfinished sessions."""

    def __init__(
        self,
        message: str,
        *,
        application_id: str,
        deployment_id: str,
        recovery_arguments: Sequence[str] = (),
        status: str = "smoke_tested",
    ) -> None:
        # Only a Release selected before can be selected again with a rollback.
        remedy = (
            "select it again with `cayu cloud rollback`"
            if status == "promoted"
            else "change `version` in cayu-cloud.toml and deploy again"
        )
        super().__init__(
            "release_superseded",
            f"{message} Cayu Cloud won't publish this Release, and the Agent keeps the one it "
            f"selected. To serve this source, {remedy}.",
        )
        self.details: dict[str, object] = {"status": status}
        app = _cloud_application_id(application_id)
        deployment = _cloud_deployment_id(deployment_id)
        if app is not None:
            self.details["application"] = app
        if deployment is not None:
            self.details["deployment_id"] = deployment
        if app is not None and deployment is not None:
            command = ["cayu", "cloud", *recovery_arguments, "deployment"]
            suffix = [deployment, "--application", app]
            commands = {
                action: shlex.join([*command, action, *suffix]) for action in ("status", "timeline")
            }
            if status == "promoted":
                commands["rollback"] = shlex.join(
                    ["cayu", "cloud", *recovery_arguments, "rollback", *suffix, "--wait"]
                )
            self.details["commands"] = commands


class _CloudSessionChoiceNotAppliedError(CloudApiError):
    """Cloud kept a more permissive session choice than the command asked for."""

    def __init__(
        self,
        not_applied: dict[str, Any],
        *,
        application_id: str,
        deployment_id: str,
        recovery_arguments: Sequence[str] = (),
    ) -> None:
        super().__init__(
            "session_choice_not_applied",
            f"{not_applied['message']} That choice may let it replace the serving release "
            "over unfinished sessions this command's choice would have kept.",
        )
        self.details: dict[str, object] = {"not_applied": not_applied}
        app = _cloud_application_id(application_id)
        deployment = _cloud_deployment_id(deployment_id)
        if app is not None:
            self.details["application"] = app
        if deployment is not None:
            self.details["deployment_id"] = deployment
        if app is not None and deployment is not None:
            command = ["cayu", "cloud", *recovery_arguments, "deployment"]
            suffix = [deployment, "--application", app]
            self.details["commands"] = {
                action: shlex.join([*command, action, *suffix]) for action in ("status", "timeline")
            }


def _report_not_applied(
    not_applied: dict[str, Any] | None,
    *,
    application_id: str,
    deployment_id: str,
    recovery_arguments: Sequence[str],
    already_selected: bool,
) -> dict[str, Any] | None:
    """Fail on a more permissive kept choice, or report what Cloud didn't apply.

    A Release Cloud selected before the command publishes under its choice whatever the
    command does, so only stderr says so; otherwise a more permissive kept choice fails.
    """

    if not_applied is None:
        return None
    if not_applied.get("more_permissive") and not already_selected:
        raise _CloudSessionChoiceNotAppliedError(
            not_applied,
            application_id=application_id,
            deployment_id=deployment_id,
            recovery_arguments=recovery_arguments,
        )
    print(f"cayu cloud: {not_applied['message']}", file=sys.stderr)
    return not_applied


class _CloudArchiveStillRunningError(CloudApiError):
    """A local wait ended while Cayu Cloud keeps retiring the archived Agent."""

    def __init__(
        self,
        *,
        application_id: str,
        operation: dict[str, Any],
        recovery_arguments: Sequence[str] = (),
    ) -> None:
        super().__init__(
            "archive_still_running",
            "Cayu Cloud is still archiving this Agent; it continues without the CLI.",
        )
        self.application_id = application_id
        self.recovery_arguments = tuple(recovery_arguments)
        self.status = str(operation.get("status"))
        blockers = operation.get("blockers")
        self.blockers = [
            {"code": str(item.get("code")), "message": str(item.get("message"))}
            for item in (blockers if isinstance(blockers, list) else [])
            if isinstance(item, dict)
        ]

    def public_details(self) -> dict[str, object]:
        return {
            "application": self.application_id,
            "blockers": self.blockers,
            "commands": {
                "status": shlex.join(
                    [
                        "cayu",
                        "cloud",
                        *self.recovery_arguments,
                        "applications",
                        "archive-status",
                        self.application_id,
                    ]
                )
            },
            "status": self.status,
        }


class _CloudServiceStillRunningError(CloudApiError):
    """A local wait ended while Cayu Cloud retained a service operation."""

    def __init__(
        self,
        *,
        application_id: str,
        deleting: bool,
        recovery_arguments: Sequence[str] = (),
        status: str,
        waited_seconds: float | None = None,
        last_issue: str | None = None,
    ) -> None:
        message = (
            "Cayu Cloud is still deleting this Agent service."
            if deleting
            else "Cayu Cloud is still starting this Agent service."
        )
        if waited_seconds is not None:
            message = f"{message[:-1]} after {waited_seconds:.0f}s."
        if last_issue is not None:
            message = f"{message} Last report: {last_issue}"
        super().__init__(
            "service_deletion_still_running" if deleting else "service_still_starting",
            message,
        )
        self.application_id = _cloud_application_id(application_id)
        self.recovery_arguments = tuple(recovery_arguments)
        self.status = status if status in _SERVICE_IN_PROGRESS else "processing"
        self.waited_seconds = waited_seconds
        self.last_issue = last_issue

    def public_details(self) -> dict[str, object]:
        details: dict[str, object] = {"status": self.status}
        if self.waited_seconds is not None:
            details["waited_seconds"] = round(self.waited_seconds)
        if self.last_issue is not None:
            details["last_issue"] = self.last_issue
        if self.application_id is not None:
            details["application"] = self.application_id
            details["commands"] = {
                "status": shlex.join(
                    [
                        "cayu",
                        "cloud",
                        *self.recovery_arguments,
                        "service",
                        "status",
                        "--application",
                        self.application_id,
                    ]
                )
            }
        return details


def _cloud_deployment_id(value: str) -> str | None:
    return value if _CLOUD_DEPLOYMENT_ID.fullmatch(value) is not None else None


def _cloud_application_id(value: str) -> str | None:
    return value if is_application_slug(value) else None


class _JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> Never:
        raise CloudCommandError("invalid_input", message)


def add_cloud_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "cloud",
        help="Manage Cayu Cloud.",
        description="Manage Cayu Cloud.",
    )
    _configure_parser(parser)


def run_cloud(arguments: argparse.Namespace) -> int:
    return _run_cloud(arguments)


def run_cloud_cli(arguments: Sequence[str]) -> int:
    try:
        parsed = _build_parser().parse_args(arguments)
    except (
        CloudApiError,
        CloudAuthError,
        CloudCommandError,
        KeyError,
        OSError,
        ValueError,
    ) as exc:
        return _cloud_failure(exc)
    return _run_cloud(parsed)


def _run_cloud(arguments: argparse.Namespace) -> int:
    try:
        result = _execute(arguments)
    except (
        CloudApiError,
        CloudAuthError,
        CloudCommandError,
        KeyError,
        OSError,
        ValueError,
    ) as exc:
        return _cloud_failure(exc)
    print(json.dumps({"ok": True, **result}, sort_keys=True))
    return 0


def _cloud_failure(exc: Exception) -> int:
    if isinstance(exc, (CloudApiError, CloudAuthError, CloudCommandError)):
        category, message = exc.category, str(exc)
    elif isinstance(exc, OSError):
        category = "local_state_unavailable"
        message = "Could not update local Cayu Cloud state."
    elif isinstance(exc, KeyError):
        category = "api_response_invalid"
        message = "Cayu Cloud API response is missing required data."
    else:
        category, message = "invalid_input", str(exc)
    error: dict[str, object] = {"category": category, "message": message}
    if isinstance(exc, CloudApiError) and exc.code is not None:
        error["code"] = exc.code
    if isinstance(exc, CloudSourceInputsError):
        error.update({"path": exc.path, "reason": exc.reason, "hint": exc.hint})
    if isinstance(
        exc,
        (
            _CloudDeploymentDiagnosticUnavailableError,
            _CloudDeploymentFailureError,
            _CloudPromotionConflictError,
            _CloudReleaseNotSelectedError,
            _CloudReleaseSupersededError,
            _CloudRetrySubmissionError,
            _CloudSessionChoiceNotAppliedError,
        ),
    ):
        error.update(exc.details)
    if isinstance(exc, _CloudDeploymentFailureError):
        error["failure"] = exc.failure
    if isinstance(exc, _CloudDeployCheckError):
        error.update(exc.details)
    if isinstance(
        exc,
        (
            _CloudArchiveStillRunningError,
            _CloudDeploymentStillRunningError,
            _CloudServiceStillRunningError,
        ),
    ):
        error.update(exc.public_details())
    if isinstance(exc, _CloudServiceHealthError):
        error["issues"] = exc.issues
    if isinstance(exc, _CloudServicePublicationError):
        error.update(exc.details)
        error["publication_error"] = exc.publication_error
    print(
        json.dumps(
            {"error": error, "ok": False},
            sort_keys=True,
        )
    )
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = _JsonArgumentParser(prog="cayu cloud", description="Manage Cayu Cloud.")
    _configure_parser(parser)
    return parser


def _configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.set_defaults(_cloud_preflight=None)
    parser.add_argument("--api-key-file", type=Path, help="Defaults to context.")
    parser.add_argument("--context", type=Path, help="Private Cayu Cloud context JSON.")
    parser.add_argument("--evidence-dir", type=Path)
    parser.add_argument("--timeout-seconds", type=_positive_finite_seconds, default=30.0)
    commands = parser.add_subparsers(dest="command", required=True)

    login = commands.add_parser(
        "login",
        help="Sign in through WorkOS.",
        description="Sign in to Cayu Cloud through WorkOS in a web browser.",
    )
    login.add_argument(
        "--no-browser",
        action="store_true",
        help="Print the verification URL and code without opening a browser.",
    )
    commands.add_parser(
        "logout",
        help="Delete the local login.",
        description="Delete the local interactive Cayu Cloud login.",
    )
    commands.add_parser(
        "whoami",
        help="Show the current user and Organization.",
        description="Show the current Cayu Cloud user and Organization.",
    )

    applications = commands.add_parser(
        "applications",
        help="List deployed Agent applications.",
        description="List Agent applications deployed in the selected Cayu Cloud.",
    )
    application_commands = applications.add_subparsers(
        dest="application_command",
        required=True,
    )
    application_list = application_commands.add_parser(
        "list",
        description="List Agent applications and their selected releases.",
    )
    application_list.add_argument(
        "--lifecycle",
        choices=("active", "archived", "all"),
        default="active",
        help="Active Agents by default; archived also lists Agents being archived.",
    )
    archive = application_commands.add_parser(
        "archive",
        description=(
            "Archive an Agent: stop its service, schedules and ingress for good and keep its "
            "Releases, configuration, data and history. Requires an organization "
            "administrator login and can't be undone."
        ),
    )
    archive.add_argument("application", type=_application_slug, help="The exact Agent slug.")
    archive.add_argument(
        "--expected-revision",
        type=int,
        required=True,
        help="The Agent revision the decision was made on (`applications archive-status`).",
    )
    archive.add_argument("--idempotency-key", help="Defaults to one derived from the revision.")
    archive.add_argument("--no-wait", action="store_true", help="Return once requested.")
    archive.add_argument("--poll-seconds", type=_positive_finite_seconds, default=5.0)
    archive.add_argument("--wait-seconds", type=_positive_finite_seconds, default=900.0)
    archive_status = application_commands.add_parser(
        "archive-status",
        description="Show an Agent's lifecycle, revision and archive progress.",
    )
    archive_status.add_argument("application", type=_application_slug)

    context = commands.add_parser(
        "context",
        help="Select or inspect a private connection context.",
        description="Select or inspect the private Cayu Cloud connection context.",
    )
    context_commands = context.add_subparsers(dest="context_command", required=True)
    context_use = context_commands.add_parser(
        "use",
        description="Select a private Cayu Cloud context file for later commands.",
    )
    context_use.add_argument("context_path", type=Path)
    context_commands.add_parser(
        "show",
        description="Show the selected context without exposing its API key.",
    )
    context_commands.add_parser(
        "clear",
        description="Forget the selected Cayu Cloud context on this machine.",
    )

    commands.add_parser(
        "doctor",
        help="Verify authentication and API access.",
        description="Verify authentication and API access to the selected Cayu Cloud.",
    )

    initialize = commands.add_parser(
        "init",
        help="Generate cayu-cloud.toml.",
        description=(
            "Generate cayu-cloud.toml from a Python Agent project. For a `cayu serve` web "
            "process, also add the cayu server extra and the environment operator auth "
            "target to pyproject.toml when they are missing; an existing auth target is "
            "never replaced."
        ),
    )
    initialize.add_argument("path", nargs="?", default=Path("."), type=Path)
    initialize.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing cayu-cloud.toml.",
    )

    deploy = commands.add_parser(
        "deploy",
        help="Publish an immutable Agent release.",
        description="Publish and activate an immutable Agent release from a local directory or repository.",
    )
    deploy.add_argument("source", nargs="?", default=".")
    deploy.add_argument("--manifest", type=Path)
    deploy.add_argument("--revision")
    deploy.add_argument(
        "--application",
        type=_application_slug,
        help=(
            "Create or update this application slug; defaults to the application "
            "declared in cayu-cloud.toml."
        ),
    )
    deploy.add_argument("--no-wait", action="store_true")
    deploy.add_argument("--no-promote", action="store_true")
    deploy.add_argument(
        "--retry-failed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Retry a replayed Cloud-side terminal failure with a fresh submission key (default: enabled).",
    )
    deploy.add_argument(
        "--skip-deploy-check",
        action="store_true",
        help=(
            "Upload even when the local `cayu check --deploy` finds problems that stop a "
            "public service starting on Cayu Cloud."
        ),
    )
    _add_acknowledge_breaking_argument(deploy)
    _add_session_choice_arguments(deploy)
    deploy.add_argument("--poll-seconds", type=_positive_finite_seconds, default=5.0)
    deploy.add_argument("--wait-seconds", type=_positive_finite_seconds, default=1800.0)
    deploy.set_defaults(_cloud_preflight=_preflight_deploy_wait)

    deployment = commands.add_parser(
        "deployment",
        help="Inspect or operate an immutable release.",
        description="Inspect, wait for, or promote one immutable Agent release.",
    )
    deployment_commands = deployment.add_subparsers(
        dest="deployment_command",
        required=True,
    )
    deployment_descriptions = {
        "logs": "Read structured publication logs for one release.",
        "status": "Show the current publication status of one release.",
        "timeline": "Show the publication milestones for one release.",
        "wait": "Wait until one release is promotable or terminal.",
        "promote": "Select a smoke-tested release for its Agent application.",
        "retry": "Create a new immutable release attempt from retained source.",
    }
    for action, description in deployment_descriptions.items():
        operation = deployment_commands.add_parser(action, description=description)
        operation.add_argument("deployment_id")
        operation.add_argument("--application", required=True)
        if action == "retry":
            operation.add_argument("--idempotency-key")
            _add_acknowledge_breaking_argument(operation)
            _add_session_choice_arguments(operation)
            operation.set_defaults(_cloud_preflight=_preflight_acknowledgements)
        if action == "logs":
            operation.add_argument("--diagnostic-offset", type=_diagnostic_offset, default=None)
            operation.add_argument("--diagnostic-limit", type=_diagnostic_limit, default=None)
        if action == "wait":
            operation.add_argument("--poll-seconds", type=_positive_finite_seconds, default=5.0)
            operation.add_argument("--wait-seconds", type=_positive_finite_seconds, default=1800.0)
            operation.set_defaults(_cloud_preflight=_preflight_wait)

    rollback = commands.add_parser(
        "rollback",
        help="Select an earlier immutable release.",
        description="Select an earlier immutable release and recreate the Agent service.",
    )
    rollback.add_argument("deployment_id")
    rollback.add_argument("--application", required=True)
    rollback.add_argument(
        "--wait",
        action="store_true",
        help=(
            "Wait until the Agent service runs the selected release, and report Cloud's "
            "publication failure if it refuses it."
        ),
    )
    _add_session_choice_arguments(rollback)
    rollback.add_argument("--poll-seconds", type=_positive_finite_seconds, default=5.0)
    rollback.add_argument("--wait-seconds", type=_positive_finite_seconds, default=1800.0)
    rollback.set_defaults(_cloud_preflight=_preflight_rollback_wait)

    runtimes = commands.add_parser(
        "runtimes",
        help="Inspect retained runtime artifacts.",
        description="Inspect retained immutable runtime artifacts for an Agent.",
    )
    runtime_commands = runtimes.add_subparsers(dest="runtime_command", required=True)
    runtime_list = runtime_commands.add_parser(
        "list",
        description="List retained runtime artifacts for an Agent application.",
    )
    runtime_list.add_argument("--application", required=True)
    runtime_status = runtime_commands.add_parser(
        "status",
        description="Show one retained runtime artifact and its lifecycle state.",
    )
    runtime_status.add_argument("artifact_id")
    runtime_status.add_argument("--application", required=True)

    service = commands.add_parser(
        "service",
        help="Operate long-horizon Agent infrastructure.",
        description="Operate the long-running web, worker, and schedule infrastructure.",
    )
    service_commands = service.add_subparsers(dest="service_command", required=True)
    service_descriptions = {
        "credentials": (
            "Show the Agent's own /cayu/ operator username and password. The output is a "
            "credential: keep it out of logs and shared terminals."
        ),
        "destroy": "Remove the Agent services and schedules but retain its releases.",
        "logs": "Read recent infrastructure logs from the Agent service.",
        "restart": "Restart the Agent service on its selected immutable release.",
        "sleep": "Scale only the Agent web service to zero.",
        "status": "Show the Agent web, worker, schedule, and endpoint state.",
        "wake": "Start only the Agent web service.",
    }
    for action, description in service_descriptions.items():
        operation = service_commands.add_parser(action, description=description)
        operation.add_argument("--application", required=True)
        if action == "destroy":
            operation.add_argument("--poll-seconds", type=_positive_finite_seconds, default=2.0)
            operation.add_argument("--wait-seconds", type=_positive_finite_seconds, default=180.0)
            operation.set_defaults(_cloud_preflight=_preflight_wait)

    environment = commands.add_parser(
        "env",
        help="Manage Agent environment variables and secrets.",
        description="Manage Agent-owned environment variables and write-only secrets.",
    )
    environment_commands = environment.add_subparsers(
        dest="environment_command",
        required=True,
    )
    environment_list = environment_commands.add_parser(
        "list",
        description="List environment variables without revealing secret values.",
    )
    environment_list.add_argument("--application", required=True)
    environment_set = environment_commands.add_parser(
        "set",
        description="Set NAME=value, or use NAME --secret --value-file PATH.",
    )
    environment_set.add_argument("assignment")
    environment_set.add_argument("--application", required=True)
    environment_set.add_argument("--secret", action="store_true")
    environment_set.add_argument(
        "--value-file",
        type=Path,
        help="Read a secret from PATH; use - for standard input.",
    )
    environment_unset = environment_commands.add_parser(
        "unset",
        description="Remove one Agent environment variable.",
    )
    environment_unset.add_argument("name")
    environment_unset.add_argument("--application", required=True)

    evidence = commands.add_parser(
        "evidence",
        help="Inspect local Cloud command evidence.",
        description="Inspect local content-free records produced by Cloud commands.",
    )
    evidence_commands = evidence.add_subparsers(dest="evidence_command", required=True)
    evidence_commands.add_parser(
        "list",
        description="List local Cayu Cloud command evidence records.",
    )
    show = evidence_commands.add_parser(
        "show",
        description="Show one local content-free evidence record.",
    )
    show.add_argument("evidence_id")
    evidence_commands.add_parser(
        "verify",
        description="Verify the integrity of every local evidence record.",
    )


def _add_acknowledge_breaking_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--acknowledge-breaking",
        action="append",
        type=_storage_revision,
        metavar="REVISION",
        help=(
            "Acknowledge a breaking Cayu storage revision the release migrates the Agent "
            "database across (repeatable). Cloud stops the previous release for the "
            "migration, and releases built with the older Cayu can't be rolled back to."
        ),
    )


def _add_session_choice_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--session-policy",
        choices=_SESSION_POLICIES,
        help=(
            "What Cayu Cloud does about the serving release's unfinished sessions before "
            "this release replaces it: wait (the default) holds publication until none "
            "blocks, block refuses at once, proceed publishes once every blocking session "
            "is acknowledged."
        ),
    )
    parser.add_argument(
        "--session-wait-seconds",
        type=_session_wait_seconds,
        metavar="SECONDS",
        help=(
            f"How long --session-policy wait holds publication ({_MIN_SESSION_WAIT_SECONDS} "
            f"to {_MAX_SESSION_WAIT_SECONDS}; Cloud's default is "
            f"{_DEFAULT_SESSION_WAIT_SECONDS})."
        ),
    )
    parser.add_argument(
        "--acknowledge-session",
        action="append",
        type=_session_id,
        metavar="SESSION_ID",
        help=(
            "Accept that this unfinished session may not resume on the new release "
            "(repeatable; '*' for every unfinished session). Implies --session-policy proceed."
        ),
    )


def _session_wait_seconds(value: str) -> int:
    if (
        re.fullmatch(r"[1-9][0-9]{0,3}", value) is None
        or not _MIN_SESSION_WAIT_SECONDS <= int(value) <= _MAX_SESSION_WAIT_SECONDS
    ):
        raise argparse.ArgumentTypeError(
            f"must be a whole number of seconds between {_MIN_SESSION_WAIT_SECONDS} and "
            f"{_MAX_SESSION_WAIT_SECONDS}"
        )
    return int(value)


def _session_id(value: str) -> str:
    if not _valid_session_id(value):
        raise argparse.ArgumentTypeError(
            f"must be a Cayu session ID of at most {_MAX_SESSION_ID_LENGTH} printable "
            f"characters, or '{_ACKNOWLEDGE_ALL_SESSIONS}'"
        )
    return value


def _valid_session_id(value: object) -> bool:
    # Cloud's rule: nonempty, no surrounding whitespace, no control characters.
    return (
        isinstance(value, str)
        and bool(value.strip())
        and value == value.strip()
        and len(value) <= _MAX_SESSION_ID_LENGTH
        and all(ord(character) >= 32 for character in value)
    )


def _session_choice(arguments: argparse.Namespace) -> dict[str, Any] | None:
    """The session choice given on the command line, as Cloud normalizes it, if any."""

    policy = getattr(arguments, "session_policy", None)
    wait_seconds = getattr(arguments, "session_wait_seconds", None)
    sessions = set(getattr(arguments, "acknowledge_session", None) or ())
    if policy is None and wait_seconds is None and not sessions:
        return None
    if len(sessions) > _MAX_SESSION_ACKNOWLEDGEMENTS:
        raise CloudCommandError(
            "invalid_input",
            f"At most {_MAX_SESSION_ACKNOWLEDGEMENTS} distinct --acknowledge-session values "
            f"can be given; use '{_ACKNOWLEDGE_ALL_SESSIONS}' to acknowledge every "
            "unfinished session.",
        )
    acknowledged = (
        [_ACKNOWLEDGE_ALL_SESSIONS] if _ACKNOWLEDGE_ALL_SESSIONS in sessions else sorted(sessions)
    )
    mode = policy or ("proceed" if acknowledged else "wait")
    if mode == "proceed" and not acknowledged:
        raise CloudCommandError(
            "invalid_input",
            "--session-policy proceed needs --acknowledge-session naming the sessions that "
            f"may not resume, or '{_ACKNOWLEDGE_ALL_SESSIONS}' for all of them.",
        )
    if mode != "proceed" and acknowledged:
        raise CloudCommandError(
            "invalid_input",
            "--acknowledge-session implies --session-policy proceed; it can't be combined "
            f"with --session-policy {mode}.",
        )
    if mode != "wait" and wait_seconds is not None:
        raise CloudCommandError(
            "invalid_input", "--session-wait-seconds applies only to --session-policy wait."
        )
    return {
        "acknowledge_sessions": acknowledged,
        "mode": mode,
        "wait_seconds": _DEFAULT_SESSION_WAIT_SECONDS if wait_seconds is None else wait_seconds,
    }


def _session_choice_payload(
    arguments: argparse.Namespace, *, omit_default: bool = False
) -> dict[str, Any] | None:
    """The request fields for the command line's session choice.

    Omitted without a choice, so the Release keeps its current one. A create request also
    omits Cloud's default choice, so it and its idempotency key stay as before.
    """

    choice = _session_choice(arguments)
    if choice is None or (omit_default and _default_session_choice(choice)):
        return None
    payload: dict[str, Any] = {"session_policy": choice["mode"]}
    if getattr(arguments, "session_wait_seconds", None) is not None:
        payload["session_wait_seconds"] = choice["wait_seconds"]
    if choice["acknowledge_sessions"]:
        payload["acknowledge_sessions"] = choice["acknowledge_sessions"]
    return payload


def _default_session_choice(choice: dict[str, Any]) -> bool:
    return choice == {
        "acknowledge_sessions": [],
        "mode": "wait",
        "wait_seconds": _DEFAULT_SESSION_WAIT_SECONDS,
    }


def _stored_session_choice(deployment: dict[str, Any]) -> dict[str, Any]:
    """A Release's session choice as Cloud returns it; absent means Cloud's default."""

    stored = deployment.get("session_policy")
    if not isinstance(stored, dict):
        return {
            "acknowledge_sessions": [],
            "mode": "wait",
            "wait_seconds": _DEFAULT_SESSION_WAIT_SECONDS,
        }
    sessions = stored.get("acknowledge_sessions")
    return {
        "acknowledge_sessions": sorted(item for item in sessions if isinstance(item, str))
        if isinstance(sessions, list)
        else sessions,
        "mode": stored.get("mode"),
        "wait_seconds": stored.get("wait_seconds"),
    }


def _stored_session_wait_seconds(release: dict[str, Any] | None) -> int:
    """How long a Release's stored choice waits; Cloud's default when it's unreadable."""

    stored = None if release is None else release.get("session_policy")
    seconds = stored.get("wait_seconds") if isinstance(stored, dict) else None
    if type(seconds) is not int or not (
        _MIN_SESSION_WAIT_SECONDS <= seconds <= _MAX_SESSION_WAIT_SECONDS
    ):
        return _DEFAULT_SESSION_WAIT_SECONDS
    return seconds


def _session_retry_acknowledgements(
    publication_error: dict[str, Any], release: dict[str, Any] | None
) -> list[str] | None:
    """The sessions a `proceed` retry of a refused Release names, or None if none can pass.

    Cloud lists only the blocking sessions the Release's choice doesn't acknowledge yet,
    and a retry replaces that choice, so the retry also names the ones it acknowledges.
    Cloud lists at most 100 sessions; when its preflight is `truncated`, a retry naming
    only those can't pass, and neither can one over Cloud's limit.
    """

    requested = publication_error.get("acknowledge_sessions")
    if not requested:
        return None
    if requested == [_ACKNOWLEDGE_ALL_SESSIONS]:
        return list(requested)
    if release is None:
        return None
    # Cloud derives the list from the preflight; without one confirming it is complete,
    # don't offer a retry.
    preflight = release.get("session_preflight")
    if not isinstance(preflight, dict) or preflight.get("truncated") is not False:
        return None
    stored = release.get("session_policy")
    acknowledged: list[str] = []
    if isinstance(stored, dict) and stored.get("mode") == "proceed":
        acknowledged = _publication_session_acknowledgements(stored.get("acknowledge_sessions"))
        if not acknowledged:
            # A proceed choice always names sessions; one that can't be read can't be kept.
            return None
    sessions = set(requested) | set(acknowledged)
    if _ACKNOWLEDGE_ALL_SESSIONS in sessions:
        return [_ACKNOWLEDGE_ALL_SESSIONS]
    if len(sessions) > _MAX_SESSION_ACKNOWLEDGEMENTS:
        return None
    return sorted(sessions)


def _storage_revision(value: str) -> int:
    if re.fullmatch(r"[1-9][0-9]{0,6}", value) is None or int(value) > _MAX_STORAGE_REVISION:
        raise argparse.ArgumentTypeError(
            f"must be a whole storage revision between 1 and {_MAX_STORAGE_REVISION}"
        )
    return int(value)


def _acknowledged_revisions(arguments: argparse.Namespace) -> tuple[int, ...]:
    """The breaking storage revisions acknowledged on the command line, as Cloud stores them."""

    revisions = tuple(sorted(set(getattr(arguments, "acknowledge_breaking", None) or ())))
    if len(revisions) > _MAX_STORAGE_ACKNOWLEDGEMENTS:
        raise CloudCommandError(
            "invalid_input",
            f"At most {_MAX_STORAGE_ACKNOWLEDGEMENTS} distinct --acknowledge-breaking "
            "revisions can be given.",
        )
    return revisions


def _acknowledgement_payload(revisions: Sequence[int]) -> dict[str, Any] | None:
    # Omitted when empty, so requests and their idempotency stay as before.
    return {"acknowledge_breaking": list(revisions)} if revisions else None


def _requested_storage_acknowledgement(publication_error: dict[str, Any]) -> tuple[int, ...]:
    """The breaking revisions Cloud's refusal asks the owner to acknowledge."""

    hint = publication_error.get("hint", "")
    listed = _ACKNOWLEDGE_BREAKING_LIST.search(hint)
    if listed is not None:
        values = [item.strip() for item in listed.group(1).split(",")]
    else:
        values = _ACKNOWLEDGE_BREAKING_FLAG.findall(hint)
    revisions: set[int] = set()
    for value in values:
        if re.fullmatch(r"[1-9][0-9]{0,6}", value) is None or int(value) > _MAX_STORAGE_REVISION:
            return ()
        revisions.add(int(value))
    if len(revisions) > _MAX_STORAGE_ACKNOWLEDGEMENTS:
        return ()
    return tuple(sorted(revisions))


def _execute(arguments: argparse.Namespace) -> dict[str, Any]:
    preflight: Callable[[argparse.Namespace], None] | None = arguments._cloud_preflight
    if preflight is not None:
        preflight(arguments)
    if arguments.command == "init":
        initialized = initialize_project(arguments.path, force=arguments.force)
        result: dict[str, Any] = {
            "application": initialized.application,
            "manifest": str(initialized.manifest_path),
            "name": initialized.name,
            "runtime": initialized.runtime,
        }
        if initialized.serve is not None:
            result["serve"] = initialized.serve
        from cayu.cli._cloud_deploy_check import init_deploy_check_notice, run_cloud_deploy_check

        check = run_cloud_deploy_check(
            initialized.manifest_path.parent,
            serves_web=initialized.runtime == "web",
            web_command=initialized.command if initialized.runtime == "web" else None,
        )
        if check.status != "not_applicable":
            result["deploy_check"] = check.public_dict()
        notice = init_deploy_check_notice(check)
        if notice is not None:
            result["warnings"] = [notice]
        return {"operation": "init", "result": result}
    if arguments.command == "login":
        return _login(arguments)
    if arguments.command == "logout":
        return {
            "operation": "logout",
            "result": {
                "signed_out": CloudAuthStore().delete(),
                "status": "signed_out",
            },
        }
    recorder = EvidenceRecorder(_evidence_directory(arguments))
    if arguments.command == "context":
        if arguments.context_command == "use":
            result = _activate_context(arguments.context_path)
        elif arguments.context_command == "show":
            result = _context_summary(arguments.context)
        else:
            result = _clear_active_context()
        return {
            "operation": f"context.{arguments.context_command}",
            "result": result,
        }
    if arguments.command == "evidence":
        if arguments.evidence_command == "list":
            return {
                "operation": "evidence.list",
                "result": {"items": recorder.list_records()},
            }
        if arguments.evidence_command == "show":
            return {
                "operation": "evidence.show",
                "result": recorder.show(arguments.evidence_id),
            }
        return {"operation": "evidence.verify", "result": recorder.verify()}
    if arguments.command == "deploy":
        recorder.preflight()
        if not arguments.no_wait:
            _validate_wait(arguments.poll_seconds, arguments.wait_seconds)
        project = resolve_project(
            arguments.source,
            manifest_path=arguments.manifest,
            revision=arguments.revision,
        )
        deploy_check = _deploy_check(project, skip=arguments.skip_deploy_check)
    if arguments.command == "doctor":
        context_path = _selected_context_path(arguments.context)
        context = _read_context(context_path) if context_path is not None else {}
        client = _cloud_client(arguments, context=context, context_path=context_path)
        applications = client.request("GET", "/v1/applications").get("items")
        if not isinstance(applications, list):
            raise CloudApiError("api_response_invalid", "Application list is invalid.")
        return {
            "operation": "doctor",
            "result": {
                "api_reachable": True,
                "api_url": client.api_url,
                "application_count": len(applications),
                "context": None if context_path is None else str(context_path),
                "deployment_id": context.get("deployment_id"),
                "region": context.get("region"),
                "status": "ready",
            },
        }
    selected_context = _selected_context_path(arguments.context)
    context = _read_context(selected_context) if selected_context is not None else {}
    client = _cloud_client(arguments, context=context, context_path=selected_context)
    if arguments.command == "whoami":
        return {
            "operation": "whoami",
            "result": client.request("GET", "/v1/me"),
        }
    if arguments.command == "applications":
        return _applications(arguments, client=client)
    if arguments.command == "deploy":
        return _deploy(
            arguments,
            client=client,
            recorder=recorder,
            project=project,
            deploy_check=deploy_check,
        )
    if arguments.command == "deployment":
        return _deployment(arguments, client=client)
    if arguments.command == "rollback":
        return _rollback(arguments, client=client)
    if arguments.command == "runtimes":
        return _runtimes(arguments, client=client)
    if arguments.command == "service":
        return _service(arguments, client=client)
    if arguments.command == "env":
        return _environment(arguments, client=client)
    raise CloudCommandError(
        "command_unavailable",
        f"`cayu cloud {arguments.command}` is not available.",
    )


def _login(arguments: argparse.Namespace) -> dict[str, Any]:
    api_url = _login_api_url(arguments)
    auth = WorkOSDeviceAuthClient(
        api_url=api_url,
        timeout_seconds=arguments.timeout_seconds,
    )
    client_id, api_hostname = auth.portal_config()
    authorization = auth.authorize_device(
        client_id=client_id,
        api_hostname=api_hostname,
    )
    print(
        f"Open {authorization.verification_uri} and enter code: {authorization.user_code}",
        file=sys.stderr,
    )
    if not arguments.no_browser:
        webbrowser.open(authorization.verification_uri_complete)
    credentials = auth.poll_for_credentials(
        authorization,
        client_id=client_id,
        api_hostname=api_hostname,
    )
    identity = CloudApiClient(
        api_url=api_url,
        api_key=credentials.access_token,
        timeout_seconds=arguments.timeout_seconds,
    ).request("GET", "/v1/me")
    if (
        identity.get("authentication") != "workos"
        or identity.get("organization_id") != credentials.organization_id
        or identity.get("user_id") != credentials.user_id
    ):
        raise CloudAuthError(
            "login_failed",
            "Cayu Cloud rejected the WorkOS login identity.",
        )
    CloudAuthStore().save(credentials)
    _clear_active_context()
    return {
        "operation": "login",
        "result": {
            "api_url": api_url,
            "authentication": "workos",
            "organization_id": credentials.organization_id,
            "status": "signed_in",
            "user_id": credentials.user_id,
        },
    }


def _login_api_url(arguments: argparse.Namespace) -> str:
    context_path = _selected_context_path(arguments.context)
    if context_path is not None and _context_source(arguments.context) != "persisted":
        return str(_read_context(context_path)["api_url"]).rstrip("/")
    return _PRODUCTION_API_URL


def _activate_context(context_path: Path) -> dict[str, Any]:
    selected = context_path.expanduser().resolve()
    context = _read_context(selected)
    _read_api_key_file(_context_file_path(selected, str(context["api_key_file"])))
    _write_private_json(
        _active_config_path(),
        {"active_context": str(selected), "schema_version": 1},
    )
    return {
        "context": str(selected),
        "deployment_id": context.get("deployment_id"),
        "region": context.get("region"),
        "status": "active",
    }


def _context_summary(explicit: Path | None) -> dict[str, Any]:
    selected, context = _selected_context(explicit)
    _read_api_key_file(_context_file_path(selected, str(context["api_key_file"])))
    return {
        "api_url": context["api_url"],
        "authentication": "file",
        "context": str(selected),
        "deployment_id": context.get("deployment_id"),
        "region": context.get("region"),
        "source": _context_source(explicit),
        "status": "active",
    }


def _clear_active_context() -> dict[str, Any]:
    config_path = _active_config_path()
    cleared = config_path.exists()
    try:
        config_path.unlink(missing_ok=True)
    except OSError as exc:
        raise CloudCommandError(
            "context_config_unavailable",
            "Could not clear the active Cayu Cloud context.",
        ) from exc
    return {
        "cleared": cleared,
        "status": "inactive",
    }


def _deploy_check(project: ResolvedCloudProject, *, skip: bool) -> dict[str, Any] | None:
    """Run the local deploy check before anything is uploaded; refuse on blocking findings."""

    if project.root is None:
        # A repository source is built from its remote revision, not this checkout.
        return None
    from cayu.cli._cloud_deploy_check import (
        DEPLOY_CHECK_ENVIRONMENT_HINT,
        deploy_check_refusal,
        run_cloud_deploy_check,
    )

    serves_web = project.manifest.web is not None
    if skip:
        return {"status": "skipped"} if serves_web else None
    check = run_cloud_deploy_check(
        project.root,
        serves_web=serves_web,
        web_command=project.manifest.web.command if project.manifest.web is not None else None,
    )
    if check.status == "not_applicable":
        return None
    if check.status == "failed":
        raise _CloudDeployCheckError(
            deploy_check_refusal(check),
            details={
                "deploy_check": check.public_dict(),
                "hint": DEPLOY_CHECK_ENVIRONMENT_HINT,
            },
        )
    return check.public_dict()


def _deploy(
    arguments: argparse.Namespace,
    *,
    client: CloudApiClient,
    recorder: EvidenceRecorder,
    project: ResolvedCloudProject,
    deploy_check: dict[str, Any] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    application = _resolve_application(
        client,
        reference=arguments.application or project.manifest.application,
        name=project.manifest.name,
    )
    repository = project.repository
    revision = project.revision
    if project.bundle is not None:
        if project.content_digest is None:
            raise CloudCommandError("source_invalid", "Local source bundle digest is missing.")
        upload = client.request(
            "POST",
            "/v1/source-bundles/uploads",
            payload={
                "content_digest": project.content_digest,
                "size_bytes": len(project.bundle),
            },
        )
        repository, revision, upload_url = _validated_source_upload(
            upload,
            content_digest=project.content_digest,
            size_bytes=len(project.bundle),
        )
        client.upload_bytes(upload_url, project.bundle)
    correlation_id = _correlation_id("deploy")
    acknowledged = _acknowledged_revisions(arguments)
    acknowledgement = _acknowledgement_payload(acknowledged)
    # The choice given on the command line, even Cloud's default, is what a retry or
    # promote of this source's Release carries. The create omits Cloud's default choice,
    # so that request and its key stay as before.
    session_choice = _session_choice(arguments)
    session_payload = _session_choice_payload(arguments)
    create_payload = {
        **(acknowledgement or {}),
        **(_session_choice_payload(arguments, omit_default=True) or {}),
    }
    retry_payload = {**(acknowledgement or {}), **(session_payload or {})} or None
    unacknowledged_payload = project.manifest.deployment_payload(
        repository=repository,
        revision=revision,
    )
    source_key = _deployment_idempotency_key(
        application_id=str(application["id"]),
        version=project.manifest.version,
        revision=revision,
        acknowledge_breaking=acknowledged,
        session_choice=session_choice,
    )
    retry: dict[str, Any] | None = None
    acknowledge_existing = False
    try:
        deployment = client.request(
            "POST",
            f"/v1/applications/{application['id']}/deployments",
            payload={**unacknowledged_payload, **create_payload},
            idempotency_key=source_key,
        )
    except CloudApiError as exc:
        if retry_payload is None or exc.status_code != 409:
            raise
        # The earlier submission may have acknowledged a different set of revisions or
        # made another session choice. Its current acknowledgement and choice can also
        # have changed through publication retries, so they cannot reconstruct the
        # original submission key. Match immutable inputs.
        existing = _existing_source_deployment(
            client,
            application_id=str(application["id"]),
            payload=unacknowledged_payload,
        )
        if existing is None:
            raise
        deployment = existing
        acknowledge_existing = True
    original = deployment
    if _deployment_status(deployment) in _DEPLOYMENT_FAILURES:
        deployment = _latest_source_retry(client, str(application["id"]), original)
    already_selected = _published_before(
        client,
        application_id=str(application["id"]),
        deployment=deployment,
        session_choice=session_choice,
    )
    if acknowledge_existing or session_choice is not None:
        # Publication may have failed on a retry of the original build. Acknowledge
        # that selected Release, while retaining the root for failed-build retries. A
        # replayed create can also return a Release whose choice a retry changed since.
        deployment, retry = _acknowledge_existing_release(
            client,
            application_id=str(application["id"]),
            deployment=deployment,
            acknowledged=acknowledged,
            session_choice=session_choice,
            session_payload=session_payload,
            source_key=source_key,
        )
    if _deployment_status(deployment) in _DEPLOYMENT_FAILURES:
        old_id = str(deployment["id"])
        old_status = str(deployment["status"])
        failure = _deployment_failure(
            client, path=f"/v1/applications/{application['id']}/deployments/{old_id}"
        )
        if (
            old_status in {"failed", "destroyed"}
            and getattr(arguments, "retry_failed", True)
            and failure is not None
            and failure["automatic_retryable"]
        ):
            # Retry the original, not the newest attempt: Cloud adds attempts as direct
            # children of the original and converges every member's retry on one live
            # attempt. The key names the failed attempt being replaced, so a repeated
            # deploy replays the same retry until that attempt fails too.
            retry_key = (
                "deploy-retry:" + hashlib.sha256(f"{source_key}:{old_id}".encode()).hexdigest()
            )
            try:
                deployment = client.request(
                    "POST",
                    f"/v1/applications/{application['id']}/deployments/{original['id']}/retry",
                    idempotency_key=retry_key,
                    # Cloud carries a Release's acknowledgement and session choice to its
                    # retries; this adds the ones given only now.
                    **({} if retry_payload is None else {"payload": retry_payload}),
                )
            except CloudApiError as exc:
                if exc.status_code == 409:
                    _raise_terminal_deployment(
                        application_id=str(application["id"]),
                        deployment_id=old_id,
                        status=old_status,
                        recovery_arguments=_cloud_recovery_arguments(arguments),
                        failure=failure,
                        replayed=True,
                        # Servers without destroyed-failure retry reject it with exactly
                        # this reason; any other conflict is the server's to explain.
                        retry_rejection=None
                        if exc.detail == _LEGACY_RETRY_REJECTION
                        else exc.detail or str(exc),
                    )
                raise _CloudRetrySubmissionError(
                    exc, deployment_id=str(original["id"]), retry_key=retry_key
                ) from exc
            retry = {
                "previous_deployment_id": old_id,
                "deployment_id": str(deployment["id"]),
                "failure_code": failure["code"],
                "reason": failure["message"],
            }
        else:
            _raise_terminal_deployment(
                application_id=str(application["id"]),
                deployment_id=old_id,
                status=old_status,
                recovery_arguments=_cloud_recovery_arguments(arguments),
                failure=failure,
                replayed=True,
            )
    # A Release Cloud promoted before this deploy, whether replayed directly or found
    # through its retry family, is not published again by any promotion. A retry this
    # deploy submitted is published by Cloud itself.
    previously_promoted = retry is None and _deployment_status(deployment) == "promoted"
    if not arguments.no_wait:
        followed_id = str(deployment["id"])

        def follow(*, until_built: bool = False) -> dict[str, Any]:
            return _wait_for_deployment(
                client,
                application_id=str(application["id"]),
                deployment_id=followed_id,
                include_failure_diagnostics=True,
                poll_seconds=arguments.poll_seconds,
                recovery_arguments=_cloud_recovery_arguments(arguments),
                wait_seconds=arguments.wait_seconds,
                sleep=sleep,
                monotonic=monotonic,
                until_built=until_built,
            )

        def apply_choice(release: dict[str, Any]) -> dict[str, Any]:
            nonlocal previously_promoted, retry
            release, applied = _acknowledge_existing_release(
                client,
                application_id=str(application["id"]),
                deployment=release,
                acknowledged=acknowledged,
                session_choice=session_choice,
                session_payload=session_payload,
                source_key=source_key,
            )
            if applied is not None:
                if retry is not None:
                    # This deploy already retried a failed build; keep what it replaced.
                    applied["previous_deployment_id"] = retry["previous_deployment_id"]
                retry = applied
                previously_promoted = False
            return release

        if _session_choice_pending(deployment, session_choice):
            # The Release was still building, or a retry returned an attempt already
            # running, so Cloud checks its publication under its earlier choice. Once it
            # is built, a retry changes a held or refused publication; otherwise the
            # promote below carries the choice.
            deployment = apply_choice(follow(until_built=True))
        deployment = follow()
        if (
            deployment.get("status") == "smoke_tested"
            and _deployment_publication_error(deployment) is not None
            and _session_choice_pending(deployment, session_choice)
        ):
            # Cloud refused it under its earlier choice after the check above.
            apply_choice(deployment)
            deployment = follow()
        if deployment.get("status") == "smoke_tested":
            # Cloud refused to publish the built Release (for example for unfinished
            # sessions) before selecting it. Promoting it would only publish it again under
            # the same choice; it waits for a retry.
            publication_error = _deployment_publication_error(deployment)
            if publication_error is not None:
                raise _CloudServicePublicationError(
                    publication_error,
                    application_id=str(application["id"]),
                    deployment_id=str(deployment["id"]),
                    recovery_arguments=_cloud_recovery_arguments(arguments),
                    release=deployment,
                )
    # A choice that goes with the promote may come after Cloud's finalize checked the
    # sessions under the earlier one, so the check it records is compared below.
    promoted_with_choice = False
    if not arguments.no_promote and deployment.get("status") == "smoke_tested":
        promoted_with_choice = _session_choice_pending(deployment, session_choice)
        application, deployment = _promote_release(
            client,
            application=application,
            deployment_id=str(deployment["id"]),
            wait=not arguments.no_wait,
            poll_seconds=arguments.poll_seconds,
            recovery_arguments=_cloud_recovery_arguments(arguments),
            wait_seconds=arguments.wait_seconds,
            sleep=sleep,
            monotonic=monotonic,
            session_payload=session_payload if promoted_with_choice else None,
        )
    runtime_artifact = None
    runtime_artifact_id = deployment.get("runtime_artifact_id")
    if isinstance(runtime_artifact_id, str) and runtime_artifact_id:
        runtime_artifact = client.request(
            "GET",
            f"/v1/applications/{application['id']}/runtime-artifacts/{runtime_artifact_id}",
        )
    service = None
    service_publication_requested = False
    if project.manifest.runtime_payload() is not None and deployment.get("status") == "promoted":
        # Unchanged source replays a Release Cloud promoted earlier. The Agent may have
        # selected another Release since, and its service may have been destroyed;
        # either way no finalizer is going to publish this Release now.
        reused = previously_promoted
        # The Agent was read when this deploy resolved it; a reused Release was promoted
        # before that, so its selection then is the one to report. Without the field
        # the selection is unknown, and the service wait below still applies.
        if (
            reused
            and "current_deployment_id" in application
            and application["current_deployment_id"] != deployment.get("id")
        ):
            raise _CloudReleaseNotSelectedError(
                application_id=str(application["id"]),
                deployment_id=str(deployment["id"]),
                current_deployment_id=application.get("current_deployment_id"),
                recovery_arguments=_cloud_recovery_arguments(arguments),
            )
        # Promotion's finalizer owns publication. A second PUT registers and
        # restarts the same processes; observe the exact release instead.
        service_absent = False
        try:
            service = _read_or_none(client, f"/v1/applications/{application['id']}/service")
        except CloudApiError as exc:
            if not _transient_api_error(exc):
                raise
            service = None
        else:
            service_absent = service is None
        # Only a confirmed 404 means nothing runs; an unavailable read is left to the
        # service wait, which tolerates brief outages.
        if service_absent and reused:
            # Nothing runs the selected Release, and its promotion finished long ago.
            # Publish it as `rollback` does; Cloud leaves a running service unchanged.
            service = client.request("PUT", f"/v1/applications/{application['id']}/service")
            service_publication_requested = True
        if not arguments.no_wait:
            service = _wait_for_service(
                client,
                application_id=str(application["id"]),
                initial=service,
                expected_deployment_id=str(deployment["id"]),
                poll_seconds=arguments.poll_seconds,
                recovery_arguments=_cloud_recovery_arguments(arguments),
                wait_seconds=arguments.wait_seconds,
                sleep=sleep,
                monotonic=monotonic,
            )
    result = {
        "application": application,
        "deployment": deployment,
        "runtime_artifact": runtime_artifact,
        "service": service,
        "service_publication_requested": service_publication_requested,
        "service_publication_pending": (
            project.manifest.runtime_payload() is not None
            and deployment.get("status") == "promoted"
            and (service is None or service.get("deployment_id") != deployment.get("id"))
        ),
        "source": {
            "content_digest": project.content_digest,
            "kind": "local_bundle" if project.bundle is not None else "repository",
            "manifest": str(project.manifest_path),
            "repository": repository,
            "revision": revision,
        },
    }
    if retry is not None:
        result["retry"] = retry
    # Without processes there's no publication for the choice to govern.
    check_preflight = promoted_with_choice and project.manifest.runtime_payload() is not None
    if check_preflight and not arguments.no_wait:
        # What Cloud recorded once the service ran this Release.
        deployment = client.request(
            "GET", f"/v1/applications/{application['id']}/deployments/{deployment['id']}"
        )
        result["deployment"] = deployment
    not_applied = _report_not_applied(
        _not_applied(
            deployment,
            session_choice=session_choice,
            acknowledge_breaking=(retry or {}).get("acknowledge_breaking_not_applied", ()),
            check_preflight=check_preflight,
        ),
        application_id=str(application["id"]),
        deployment_id=str(deployment["id"]),
        recovery_arguments=_cloud_recovery_arguments(arguments),
        already_selected=already_selected,
    )
    if not_applied is not None:
        result["not_applied"] = not_applied
    if deploy_check is not None:
        result["deploy_check"] = deploy_check
    safe_result = recorder.redact(result)
    response: dict[str, Any] = {
        "evidence_id": None,
        "operation": "deploy",
        "result": safe_result,
    }
    try:
        evidence_path = recorder.record(
            "deploy",
            result=safe_result,
            correlation_id=correlation_id,
        )
    except OSError:
        response["evidence"] = {
            "category": "local_state_unavailable",
            "message": "Deployment succeeded, but local evidence could not be recorded.",
            "status": "unavailable",
        }
    else:
        response["evidence_id"] = evidence_path.name
    return response


def _existing_source_deployment(
    client: CloudApiClient,
    *,
    application_id: str,
    payload: dict[str, Any],
) -> dict[str, Any] | None:
    """Find the original Release only when every immutable submission input matches."""

    # Cloud's canonical manifest omits an absent runtime, and includes a web process's
    # unset idle timeout as null. Other optional defaults are already omitted by the
    # project serializer, just as Cloud omits them to preserve existing digests.
    manifest = dict(payload["manifest"])
    runtime = manifest.get("runtime")
    if runtime is None:
        manifest.pop("runtime", None)
    elif runtime.get("web") is not None:
        manifest["runtime"] = {
            **runtime,
            "web": {"idle_timeout_seconds": None, **runtime["web"]},
        }
    digest = hashlib.sha256(
        json.dumps(manifest, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    cursor = None
    seen = set()
    for _page in range(100):
        query = {"limit": "100"}
        if cursor is not None:
            query["cursor"] = cursor
        page = client.request("GET", f"/v1/applications/{application_id}/deployments", query=query)
        items = page.get("items")
        if not isinstance(items, list):
            raise CloudApiError(
                "api_response_invalid", "Cayu Cloud returned an invalid deployment list."
            )
        for item in items:
            if (
                isinstance(item, dict)
                and item.get("version") == payload["version"]
                and item.get("manifest_digest") == digest
                and item.get("policy_version") == payload["policy_version"]
            ):
                return item
        cursor = page.get("next_cursor")
        if cursor is None:
            return None
        if not isinstance(cursor, str) or cursor in seen:
            break
        seen.add(cursor)
    raise CloudApiError(
        "api_response_invalid", "Cayu Cloud deployment history pagination did not converge."
    )


def _acknowledge_existing_release(
    client: CloudApiClient,
    *,
    application_id: str,
    deployment: dict[str, Any],
    acknowledged: Sequence[int],
    source_key: str,
    session_choice: dict[str, Any] | None = None,
    session_payload: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Add the acknowledgement and session choice to an earlier submission of the same source.

    Cloud adds them when the Release's publication failed and awaits a retry, and then
    publishes the same Release again. A publication waiting for sessions takes only the
    session choice; Cloud ignores an acknowledgement there, so it isn't sent and the
    result says it wasn't applied. A failed build is retried by the caller with them; a
    Release still in progress is followed as it is.
    """

    existing = deployment.get("acknowledge_breaking")
    needs_acknowledgement = bool(acknowledged) and not (
        isinstance(existing, list) and set(acknowledged) <= set(existing)
    )
    needs_choice = _session_choice_pending(deployment, session_choice)
    if not needs_acknowledgement and not needs_choice:
        return deployment, None
    if _deployment_status(deployment) in _DEPLOYMENT_FAILURES:
        return deployment, None
    deployment_id = str(deployment["id"])
    path = f"/v1/applications/{application_id}/deployments/{deployment_id}"
    current = client.request("GET", path)
    publication_error = _deployment_publication_error(current)
    waiting = None if not needs_choice else _waiting_session_preflight(current)
    if publication_error is None and waiting is None:
        # The fresh read carries the session check Cloud recorded so far.
        return (current if needs_choice else deployment), None
    retried = client.request(
        "POST",
        f"{path}/retry",
        idempotency_key="deploy-acknowledge:"
        + hashlib.sha256(f"{source_key}:{deployment_id}".encode()).hexdigest(),
        payload={
            **(
                (_acknowledgement_payload(acknowledged) or {})
                if publication_error is not None
                else {}
            ),
            **(session_payload or {}),
        },
    )
    retry: dict[str, Any] = {
        "deployment_id": str(retried["id"]),
        "previous_deployment_id": deployment_id,
    }
    if acknowledged:
        if publication_error is None and needs_acknowledgement:
            retry["acknowledge_breaking_not_applied"] = list(acknowledged)
        else:
            retry["acknowledge_breaking"] = list(acknowledged)
    if session_choice is not None:
        retry["session_policy"] = session_choice
    if publication_error is not None:
        retry["failure_code"] = publication_error["code"]
        retry["reason"] = publication_error["message"]
    elif waiting is not None:
        retry["reason"] = waiting["message"]
    return retried, retry


def _published_before(
    client: CloudApiClient,
    *,
    application_id: str,
    deployment: dict[str, Any],
    session_choice: dict[str, Any] | None,
) -> bool:
    """Whether a Release this command found was already past its session check.

    Then whatever choice it keeps isn't this command's doing. A `promoted` Release may
    still be mid-check, so it counts only when its recorded check already passed or the
    Agent's service runs it; an unreadable service counts as not running it. Only read
    when the command's session choice differs from the Release's.
    """

    if _deployment_status(deployment) != "promoted" or not _session_choice_pending(
        deployment, session_choice
    ):
        return False
    preflight = deployment.get("session_preflight")
    if isinstance(preflight, dict) and preflight.get("state") in _PASSED_SESSION_CHECKS:
        return True
    try:
        service = _read_or_none(client, f"/v1/applications/{application_id}/service")
    except CloudApiError as exc:
        if not _transient_api_error(exc):
            raise
        return False
    return service is not None and service.get("deployment_id") == deployment.get("id")


def _session_choice_pending(deployment: dict[str, Any], choice: dict[str, Any] | None) -> bool:
    """Whether the command line made a session choice the Release doesn't have."""

    return choice is not None and _stored_session_choice(deployment) != choice


def _not_applied(
    deployment: dict[str, Any],
    *,
    session_choice: dict[str, Any] | None,
    acknowledge_breaking: Sequence[int] = (),
    check_preflight: bool = False,
) -> dict[str, Any] | None:
    """What the command line asked for that the Release doesn't carry, if anything.

    Cloud takes a session choice only for a new Release or attempt, a publication that
    waits or was refused, or a promotion, and an acknowledgement only for a refused or
    paused publication or a new attempt. A command that comes later, or finds an attempt
    already running, leaves the Release as it was.

    A choice sent with a promote is stored even when Cloud's own finalize already passed
    the session check under the earlier one, so `check_preflight` also compares the
    Release's recorded check with the choice. Without a recorded check the choice is
    reported as unconfirmed.
    """

    carried = deployment.get("acknowledge_breaking")
    missing = sorted(set(acknowledge_breaking) - set(carried if isinstance(carried, list) else ()))
    release = _cloud_deployment_id(str(deployment.get("id")))
    subject = "the Release" if release is None else f"Release {release}"
    result: dict[str, Any] = {}
    parts = []
    if session_choice is not None:
        stored = _stored_session_choice(deployment)
        outcome, checked = "applied", None
        if stored != session_choice:
            outcome = "not_applied"
        elif check_preflight:
            outcome, checked = _checked_session_choice(deployment, session_choice)
        if outcome != "applied":
            kept = checked or stored
            result.update(
                {
                    "confirmed": outcome == "not_applied",
                    "more_permissive": _more_permissive_publication(
                        deployment, stored, session_choice
                    ),
                    "release_session_policy": stored,
                    "session_policy": session_choice,
                }
            )
            earlier = f"`{kept['mode']}` " if kept.get("mode") in _SESSION_POLICIES else ""
            if outcome == "unconfirmed":
                parts.append(
                    f"Cayu Cloud has recorded no session check for {subject} under this "
                    "command's session choice, so it isn't confirmed that publication used it."
                )
            else:
                if checked is not None:
                    result["checked_session_policy"] = checked
                parts.append(
                    f"Cayu Cloud did not apply this command's session choice to {subject}, "
                    + (
                        f"which passed its session check under the earlier {earlier}choice."
                        if checked is not None
                        else f"which keeps its earlier {earlier}choice."
                    )
                )
    if missing:
        result["acknowledge_breaking"] = missing
        parts.append(
            f"Cayu Cloud did not apply --acknowledge-breaking "
            f"{', '.join(str(revision) for revision in missing)} to {subject}; if it refuses "
            "the breaking migration later, its error gives the retry command."
        )
    if not parts:
        return None
    result["message"] = " ".join(parts)
    return result


def _checked_session_choice(
    deployment: dict[str, Any], requested: dict[str, Any]
) -> tuple[str, dict[str, Any] | None]:
    """Whether the Release's recorded session check agrees with the requested choice.

    Returns `applied`, `unconfirmed`, or `not_applied` with the choice the check
    effectively passed under. A check that found nothing blocking (or couldn't read
    sessions, which publishes under any choice) agrees with every choice. One that
    published by acknowledging sessions agrees only with a choice acknowledging them too;
    its per-session `acknowledged` flags say which, unless the list was truncated.
    """

    preflight = deployment.get("session_preflight")
    if not isinstance(preflight, dict):
        return "unconfirmed", None
    state = preflight.get("state")
    if state in {"clear", "unsupported", "not_running", "superseded"}:
        return "applied", None
    if state in {"waiting", "blocked"}:
        # Cloud rechecks a waiting or refused publication under the stored choice.
        return ("applied" if preflight.get("policy") == requested["mode"] else "unconfirmed"), None
    if state != "acknowledged":
        return "unconfirmed", None
    if _ACKNOWLEDGE_ALL_SESSIONS in requested["acknowledge_sessions"]:
        return "applied", None
    if preflight.get("read") != "complete":
        # Unread sessions pass only under '*'.
        acknowledged = [_ACKNOWLEDGE_ALL_SESSIONS]
    else:
        sessions = preflight.get("sessions")
        if not isinstance(sessions, list) or preflight.get("truncated") is not False:
            return "unconfirmed", None
        acknowledged = sorted(
            {
                item["id"]
                for item in sessions
                if isinstance(item, dict)
                and item.get("blocking") is True
                and item.get("acknowledged") is True
                and _valid_session_id(item.get("id"))
            }
        )
    checked = {"acknowledge_sessions": acknowledged, "mode": "proceed"}
    if _more_permissive_session_choice(checked, requested):
        return "not_applied", checked
    return "applied", None


def _more_permissive_publication(
    deployment: dict[str, Any], kept: dict[str, Any], requested: dict[str, Any]
) -> bool:
    """Whether the Release can publish over sessions the requested choice would protect.

    A recorded session check decides: one that found nothing blocking, or refused, can't
    strand a session, and one that acknowledged sessions can only if the requested
    choice doesn't acknowledge them (or its list was truncated, or its state is
    unknown). Without a check yet, or while one holds publication, Cloud decides later
    under the kept choice.
    """

    preflight = deployment.get("session_preflight")
    state = preflight.get("state") if isinstance(preflight, dict) else None
    if state is None or state == "waiting":
        return _more_permissive_session_choice(kept, requested)
    if state == "blocked":
        return False
    return _checked_session_choice(deployment, requested)[0] != "applied"


def _more_permissive_session_choice(kept: dict[str, Any], requested: dict[str, Any]) -> bool:
    """Whether `kept` publishes over sessions that `requested` would not.

    `block` and `wait` never publish over a blocking session; they differ only in when
    they refuse. `proceed` does, and between two `proceed` choices one acknowledging a
    session the other doesn't (or '*') is more permissive.
    """

    if _ACKNOWLEDGE_ALL_SESSIONS in requested["acknowledge_sessions"]:
        return False
    order = {"block": 0, "wait": 0, "proceed": 1}
    mode = kept.get("mode")
    if mode not in order:
        return True
    if order[mode] != order[requested["mode"]]:
        return order[mode] > order[requested["mode"]]
    if mode != "proceed":
        return False
    sessions = kept.get("acknowledge_sessions")
    if not isinstance(sessions, list):
        return True
    return not set(sessions) <= set(requested["acknowledge_sessions"])


def _deployment_idempotency_key(
    *,
    application_id: str,
    version: str,
    revision: str,
    acknowledge_breaking: Sequence[int] = (),
    session_choice: dict[str, Any] | None = None,
) -> str:
    # Cloud refuses a key reused with another acknowledgement or session choice, so a
    # nonempty acknowledgement and a non-default choice are part of the key. Without
    # them the key is unchanged from earlier CLI releases.
    identity = json.dumps(
        [application_id, version, revision]
        + ([sorted(set(acknowledge_breaking))] if acknowledge_breaking else [])
        + (
            [{"session_policy": session_choice}]
            if session_choice is not None and not _default_session_choice(session_choice)
            else []
        ),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return "deploy:" + hashlib.sha256(identity).hexdigest()


def _validated_source_upload(
    payload: dict[str, Any],
    *,
    content_digest: str,
    size_bytes: int,
) -> tuple[str, str, str]:
    repository = payload.get("repository")
    revision = payload.get("revision")
    upload_url = payload.get("upload_url")
    if (
        payload.get("content_digest") != content_digest
        or payload.get("size_bytes") != size_bytes
        or not isinstance(repository, str)
        or repository
        != ("cayu-cloud://source-bundles/sha256/" + content_digest.removeprefix("sha256:"))
        or not isinstance(revision, str)
        or revision != content_digest.removeprefix("sha256:")[:40]
        or not isinstance(upload_url, str)
        or not upload_url
    ):
        raise CloudApiError(
            "api_response_invalid",
            "Cayu Cloud returned an invalid local source upload.",
        )
    return repository, revision, upload_url


def _resolve_application(
    client: CloudApiClient,
    *,
    reference: str,
    name: str,
) -> dict[str, Any]:
    reference = _application_slug(reference)
    applications = client.request("GET", "/v1/applications").get("items", [])
    if not isinstance(applications, list):
        raise CloudApiError("api_response_invalid", "Application list is invalid.")
    matches = [
        application
        for application in applications
        if isinstance(application, dict) and application.get("id") == reference
    ]
    if len(matches) > 1:
        raise CloudApiError("application_ambiguous", "Application reference is ambiguous.")
    if matches:
        return matches[0]
    return client.request(
        "PUT",
        f"/v1/applications/{reference}",
        payload={"name": name},
    )


def _environment(
    arguments: argparse.Namespace,
    *,
    client: CloudApiClient,
) -> dict[str, Any]:
    if (
        arguments.environment_command == "set"
        and arguments.secret
        and ("=" in str(arguments.assignment) or arguments.value_file is None)
    ):
        raise CloudCommandError(
            "secret_value_source_required",
            "Secret values must use NAME --secret --value-file PATH; never put them in argv.",
        )
    application_id = _application_id(client, arguments.application)
    base = f"/v1/applications/{application_id}/environment"
    if arguments.environment_command == "list":
        return {
            "operation": "env.list",
            "result": client.request("GET", base),
        }
    if arguments.environment_command == "unset":
        name = _environment_name(arguments.name)
        result = client.request("DELETE", f"{base}/{name}")
        return {
            "operation": "env.unset",
            "result": result,
        }
    assignment = str(arguments.assignment)
    if arguments.secret:
        name = _environment_name(assignment)
        value = _read_secret_value(arguments.value_file)
    else:
        if arguments.value_file is not None or "=" not in assignment:
            raise CloudCommandError(
                "environment_assignment_invalid",
                "Plain values must use NAME=value; --value-file is reserved for secrets.",
            )
        raw_name, value = assignment.split("=", 1)
        name = _environment_name(raw_name)
    result = client.request(
        "PUT",
        f"{base}/{name}",
        payload={"secret": bool(arguments.secret), "value": value},
    )
    return {
        "operation": "env.set",
        "result": result,
    }


def _environment_name(value: str) -> str:
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value) is None:
        raise CloudCommandError(
            "environment_name_invalid",
            "Environment variable names must use letters, numbers, and underscores.",
        )
    return value


def _read_secret_value(path: Path) -> str:
    try:
        value = sys.stdin.read() if str(path) == "-" else path.expanduser().read_text()
    except OSError as exc:
        raise CloudCommandError(
            "secret_value_unavailable",
            "Could not read the secret value file.",
        ) from exc
    if value.endswith("\r\n"):
        value = value[:-2]
    elif value.endswith("\n"):
        value = value[:-1]
    if not value:
        raise CloudCommandError("secret_value_empty", "Secret values cannot be empty.")
    if len(value) > 65_536:
        raise CloudCommandError("secret_value_too_large", "Secret values are limited to 64 KiB.")
    return value


def _validated_deployment_diagnostics(result: dict[str, Any]) -> dict[str, Any]:
    result = dict(result)
    failure = result.get("failure")
    if isinstance(failure, dict) and "schema_version" in failure:
        result["failure"] = parse_build_failure(failure)
        if result["failure"] is None:
            result["diagnostic_status"] = "unavailable_or_unsupported"
    if "diagnostics" in result:
        candidates = result["diagnostics"]
        if not isinstance(candidates, list) or len(candidates) > 100:
            candidates = []
            result["diagnostic_status"] = "unavailable_or_unsupported"
        diagnostics = [parse_build_failure(candidate) for candidate in candidates]
        if any(diagnostic is None for diagnostic in diagnostics):
            result["diagnostic_status"] = "unavailable_or_unsupported"
        result["diagnostics"] = [item for item in diagnostics if item is not None]
        offset = result.get("next_diagnostic_offset")
        if offset is not None and (type(offset) is not int or offset < 0):
            result["next_diagnostic_offset"] = None
    return result


def _diagnostic_offset(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("diagnostic offset must be nonnegative")
    return number


def _diagnostic_limit(value: str) -> int:
    number = int(value)
    if not 1 <= number <= 100:
        raise argparse.ArgumentTypeError("diagnostic limit must be between 1 and 100")
    return number


def _deployment(
    arguments: argparse.Namespace,
    *,
    client: CloudApiClient,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    if arguments.deployment_command == "wait":
        _validate_wait(arguments.poll_seconds, arguments.wait_seconds)
    application_id = _application_id(client, arguments.application)
    base = f"/v1/applications/{application_id}/deployments/{arguments.deployment_id}"
    if arguments.deployment_command == "status":
        result = client.request("GET", base)
        _deployment_status(result)
    elif arguments.deployment_command in {"logs", "timeline"}:
        query: dict[str, str] = {}
        if arguments.deployment_command == "logs":
            for name in ("diagnostic_offset", "diagnostic_limit"):
                value = getattr(arguments, name, None)
                if value is not None:
                    query[name] = str(value)
        result = client.request(
            "GET", f"{base}/{arguments.deployment_command}", query=query or None
        )
        result = _validated_deployment_diagnostics(result)

    elif arguments.deployment_command == "retry":
        acknowledged = _acknowledged_revisions(arguments)
        payload = {
            **(_acknowledgement_payload(acknowledged) or {}),
            **(_session_choice_payload(arguments) or {}),
        } or None
        try:
            result = client.request(
                "POST",
                base + "/retry",
                idempotency_key=arguments.idempotency_key or _correlation_id("deployment-retry"),
                **({} if payload is None else {"payload": payload}),
            )
        except CloudApiError as exc:
            if exc.status_code != 409 or exc.detail is None:
                raise
            raise CloudApiError(
                exc.category,
                f"Cayu Cloud API returned HTTP 409: {exc.detail}",
                status_code=409,
                detail=exc.detail,
            ) from exc
        # A retry of a publication waiting for sessions takes only the session choice, and
        # one that finds an attempt already running takes neither.
        not_applied = _report_not_applied(
            _not_applied(
                result,
                session_choice=_session_choice(arguments),
                acknowledge_breaking=acknowledged,
            ),
            application_id=application_id,
            deployment_id=arguments.deployment_id,
            recovery_arguments=_cloud_recovery_arguments(arguments),
            # A promoted Release was selected before this retry; one still in progress
            # publishes under the choice it keeps.
            already_selected=_deployment_status(result) == "promoted",
        )
        if not_applied is not None:
            return {
                "not_applied": not_applied,
                "operation": "deployment.retry",
                "result": result,
            }
    elif arguments.deployment_command == "wait":
        result = _wait_for_deployment(
            client,
            application_id=application_id,
            deployment_id=arguments.deployment_id,
            include_failure_diagnostics=True,
            poll_seconds=arguments.poll_seconds,
            recovery_arguments=_cloud_recovery_arguments(arguments),
            wait_seconds=arguments.wait_seconds,
            sleep=sleep,
            monotonic=monotonic,
        )
        # A ready Release whose service publication failed (for example a refused
        # breaking storage migration) waits for a retry, not for more time.
        publication_error = _deployment_publication_error(result)
        if publication_error is not None:
            raise _CloudServicePublicationError(
                publication_error,
                application_id=application_id,
                deployment_id=arguments.deployment_id,
                recovery_arguments=_cloud_recovery_arguments(arguments),
                release=result,
            )
    else:
        application = client.request("GET", f"/v1/applications/{application_id}")
        result = client.request(
            "POST",
            base + "/promote",
            payload={"expected_application_revision": application["revision"]},
        )
    return {
        "operation": f"deployment.{arguments.deployment_command}",
        "result": result,
    }


def _rollback(
    arguments: argparse.Namespace,
    *,
    client: CloudApiClient,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    application_id = _application_id(client, arguments.application)
    application = client.request("GET", f"/v1/applications/{application_id}")
    selected = client.request(
        "POST",
        (f"/v1/applications/{application_id}/deployments/{arguments.deployment_id}/rollback"),
        payload={
            "expected_application_revision": application["revision"],
            # Cloud checks the serving release's sessions before it publishes this one.
            **(_session_choice_payload(arguments) or {}),
        },
    )
    service = client.request(
        "PUT",
        f"/v1/applications/{application_id}/service",
    )
    if getattr(arguments, "wait", False):
        # Cloud publishes the selected release asynchronously and may refuse it, for
        # example when the database was migrated past what its Cayu supports.
        service = _wait_for_service(
            client,
            application_id=application_id,
            initial=None,
            expected_deployment_id=arguments.deployment_id,
            poll_seconds=arguments.poll_seconds,
            recovery_arguments=_cloud_recovery_arguments(arguments),
            wait_seconds=arguments.wait_seconds,
            sleep=sleep,
            monotonic=monotonic,
        )
    return {
        "operation": "rollback",
        "result": {"application": selected, "service": service},
    }


def _runtimes(
    arguments: argparse.Namespace,
    *,
    client: CloudApiClient,
) -> dict[str, Any]:
    application_id = _application_id(client, arguments.application)
    collection = f"/v1/applications/{application_id}/runtime-artifacts"
    if arguments.runtime_command == "list":
        result = client.request("GET", collection)
    else:
        result = client.request("GET", f"{collection}/{arguments.artifact_id}")
    return {
        "operation": f"runtimes.{arguments.runtime_command}",
        "result": result,
    }


def _service(
    arguments: argparse.Namespace,
    *,
    client: CloudApiClient,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    action = arguments.service_command
    if action == "destroy":
        _validate_wait(arguments.poll_seconds, arguments.wait_seconds)
    application_id = _application_id(client, arguments.application)
    path = f"/v1/applications/{application_id}/service"
    if action == "status":
        result = client.request("GET", path)
        _service_status(result)
    elif action == "restart":
        result = client.request("POST", f"{path}/restart")
    elif action in {"sleep", "wake"}:
        result = client.request("POST", f"{path}/{action}")
    elif action == "logs":
        result = client.request("GET", f"{path}/logs")
    elif action == "credentials":
        result = _operator_credentials(client, application_id=application_id)
    else:
        result = client.request("DELETE", path)
        result_status = None if not result else _service_status(result)
        if result_status == "deleting":
            _validate_wait(arguments.poll_seconds, arguments.wait_seconds)
            deadline = monotonic() + arguments.wait_seconds
            while result_status == "deleting":
                if monotonic() >= deadline:
                    raise _CloudServiceStillRunningError(
                        application_id=application_id,
                        deleting=True,
                        recovery_arguments=_cloud_recovery_arguments(arguments),
                        status=result_status,
                    )
                sleep(arguments.poll_seconds)
                result = client.request("GET", path)
                result_status = _service_status(result)
    return {"operation": f"service.{action}", "result": result}


def _operator_credentials(client: CloudApiClient, *, application_id: str) -> dict[str, Any]:
    """The Agent's per-Agent `/cayu/` login; printed to stdout only, never recorded."""

    try:
        response = client.request("GET", f"/v1/applications/{application_id}/operator-credentials")
    except CloudApiError as exc:
        # The Agent was just resolved, so a 404 here means a Cloud without this endpoint.
        if exc.status_code == 404:
            raise CloudCommandError(
                "operator_credentials_unsupported",
                "This Cayu Cloud does not support per-Agent operator credentials yet.",
            ) from None
        raise
    username = response.get("username")
    password = response.get("password")
    environment = response.get("env")
    if (
        not isinstance(username, str)
        or not username
        or not isinstance(password, str)
        or not password
        or not isinstance(environment, dict)
        or not isinstance(environment.get("username"), str)
        or not isinstance(environment.get("password"), str)
    ):
        raise CloudApiError("api_response_invalid", "Operator credentials response is invalid.")
    return {
        "env": {"password": environment["password"], "username": environment["username"]},
        "password": password,
        "username": username,
    }


def _applications(
    arguments: argparse.Namespace,
    *,
    client: CloudApiClient,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    action = arguments.application_command
    if action == "list":
        return {
            "operation": "applications.list",
            # The server lists active Agents by default; send only a different filter.
            "result": client.request(
                "GET",
                "/v1/applications",
                query=None
                if arguments.lifecycle == "active"
                else {"lifecycle": arguments.lifecycle},
            ),
        }
    application_id = arguments.application
    path = f"/v1/applications/{application_id}"
    if action == "archive-status":
        return {
            "operation": "applications.archive-status",
            "result": {
                "application": client.request("GET", path),
                "archive": _read_or_none(client, f"{path}/archive"),
            },
        }
    if not arguments.no_wait:
        _validate_wait(arguments.poll_seconds, arguments.wait_seconds)
    # One decision, one key: retrying the same revision replays the same archive.
    idempotency_key = (
        arguments.idempotency_key or f"archive:{application_id}:{arguments.expected_revision}"
    )
    operation = client.request(
        "POST",
        f"{path}/archive",
        payload={"expected_application_revision": arguments.expected_revision},
        idempotency_key=idempotency_key,
    )
    if not arguments.no_wait:
        deadline = monotonic() + arguments.wait_seconds
        transient_failures = 0
        while _archive_status(operation) != "completed":
            if monotonic() >= deadline:
                raise _CloudArchiveStillRunningError(
                    application_id=application_id,
                    operation=operation,
                    recovery_arguments=_cloud_recovery_arguments(arguments),
                )
            sleep(arguments.poll_seconds)
            try:
                operation = client.request("GET", f"{path}/archive")
                transient_failures = 0
            except CloudApiError as exc:
                # Archive keeps running in Cayu Cloud: a brief API outage is not a failure.
                if (
                    not _transient_api_error(exc)
                    or transient_failures >= _SERVICE_POLL_TRANSIENT_LIMIT
                ):
                    raise
                transient_failures += 1
    return {"operation": "applications.archive", "result": operation}


def _archive_status(operation: dict[str, Any]) -> str:
    status = operation.get("status")
    if status not in {"requested", "in_progress", "blocked", "completed"}:
        raise CloudApiError("api_response_invalid", "Archive status is invalid.")
    return str(status)


def _application_id(client: CloudApiClient, reference: str) -> str:
    # Archived Agents stay readable (status, logs, history), so resolve across lifecycles.
    applications = client.request("GET", "/v1/applications", query={"lifecycle": "all"}).get(
        "items", []
    )
    if not isinstance(applications, list):
        raise CloudApiError("api_response_invalid", "Application list is invalid.")
    candidates = [application for application in applications if isinstance(application, dict)]
    # An exact slug always names its Agent: an archived Agent's reserved display name must
    # never make an active Agent's own slug ambiguous.
    matches = [application for application in candidates if application.get("id") == reference]
    if not matches:
        matches = [
            application
            for application in candidates
            if application.get("name") == reference
            or _slug(str(application.get("name", ""))) == reference
        ]
    if len(matches) != 1:
        raise CloudApiError(
            "application_not_found" if not matches else "application_ambiguous",
            f"Application reference did not resolve exactly once: {reference}",
        )
    return str(matches[0]["id"])


def _wait_for_deployment(
    client: CloudApiClient,
    *,
    application_id: str,
    deployment_id: str,
    include_failure_diagnostics: bool = False,
    poll_seconds: float,
    recovery_arguments: Sequence[str] = (),
    wait_seconds: float,
    sleep: Callable[[float], None],
    monotonic: Callable[[], float],
    until_built: bool = False,
) -> dict[str, Any]:
    """Wait until the Release is ready, or only until it is built with `until_built`."""

    _validate_wait(poll_seconds, wait_seconds)
    deadline = monotonic() + wait_seconds
    path = f"/v1/applications/{application_id}/deployments/{deployment_id}"
    hold = _SessionHoldWatch(wait_seconds=wait_seconds, monotonic=monotonic)
    while True:
        deployment = client.request("GET", path)
        status = _deployment_status(deployment)
        _raise_if_superseded(
            deployment,
            application_id=application_id,
            deployment_id=deployment_id,
            recovery_arguments=recovery_arguments,
            client=client,
        )
        # A built Release whose publication waits for the serving release's sessions is
        # not ready yet: Cloud selects and publishes it, or refuses it, once they settle.
        deadline = hold.observe(deployment, deadline)
        if status in _DEPLOYMENT_READY and (hold.waiting is None or until_built):
            return deployment
        if status in _DEPLOYMENT_FAILURES:
            _raise_terminal_deployment(
                application_id=application_id,
                deployment_id=deployment_id,
                status=status,
                recovery_arguments=recovery_arguments,
                failure=_deployment_failure(client, path=path)
                if include_failure_diagnostics
                else None,
            )
        if monotonic() >= deadline:
            raise _CloudDeploymentStillRunningError(
                application_id=application_id,
                deployment_id=deployment_id,
                recovery_arguments=recovery_arguments,
                status=status,
                last_issue=hold.waiting,
            )
        sleep(poll_seconds)


class _SessionHoldWatch:
    """Follow a Release's publication while Cloud holds it for unfinished sessions.

    Cloud keeps the Release's status while its `session_preflight` waits, until the
    preflight's `deadline_at`. Each new waiting preflight is reported on standard error
    and extends the local wait to that deadline plus the command's own wait, so the
    publication that follows has its usual time.
    """

    def __init__(self, *, wait_seconds: float, monotonic: Callable[[], float]) -> None:
        self.wait_seconds = wait_seconds
        self.monotonic = monotonic
        self.waiting: str | None = None
        self._seen: tuple[object, ...] | None = None
        self._reported: tuple[object, ...] | None = None

    def observe(self, deployment: dict[str, Any], deadline: float) -> float:
        preflight = _waiting_session_preflight(deployment)
        if preflight is None:
            self.waiting = None
            return deadline
        self.waiting = preflight["message"]
        report = (preflight["message"], preflight["waiting_for"], preflight["deadline_at"])
        if report != self._reported:
            self._reported = report
            until = (
                f" Cayu Cloud decides by {preflight['deadline_at']}."
                if preflight["deadline_at"] is not None
                else ""
            )
            print(f"cayu cloud: {preflight['message']}{until}", file=sys.stderr)
        # Only a new check extends the wait, so a preflight that stops changing can't
        # hold the command forever.
        seen = (preflight["checked_at"], *report)
        if seen == self._seen or preflight["remaining_seconds"] is None:
            return deadline
        self._seen = seen
        return max(deadline, self.monotonic() + preflight["remaining_seconds"] + self.wait_seconds)


def _waiting_session_preflight(deployment: dict[str, Any]) -> dict[str, Any] | None:
    """The Release's validated session preflight while it holds publication, else None.

    Unknown fields are ignored. A missing or invalid deadline still reports the wait but
    does not extend it.
    """

    candidate = deployment.get("session_preflight")
    if not isinstance(candidate, dict) or candidate.get("state") != "waiting":
        return None
    message = _deployment_failure_string(candidate.get("message"), max_bytes=4096)
    if message is None or safe_text(message, 4096) is None:
        message = "Cayu Cloud is waiting for the serving release's unfinished sessions."
    waiting_for = candidate.get("waiting_for")
    if not isinstance(waiting_for, str) or _SESSION_WAITING_FOR.fullmatch(waiting_for) is None:
        waiting_for = None
    deadline = _cloud_timestamp(candidate.get("deadline_at"))
    checked = _cloud_timestamp(candidate.get("checked_at"))
    remaining = None
    if deadline is not None:
        # Measured on Cloud's clock when it checked, so local clock skew doesn't matter.
        reference = checked if checked is not None else datetime.now(UTC)
        remaining = min(
            max((deadline - reference).total_seconds(), 0.0), float(_MAX_SESSION_WAIT_SECONDS)
        )
    return {
        "checked_at": None if checked is None else candidate["checked_at"],
        "deadline_at": None if deadline is None else candidate["deadline_at"],
        "message": message,
        "remaining_seconds": remaining,
        "waiting_for": waiting_for,
    }


def _raise_if_superseded(
    deployment: dict[str, Any],
    *,
    application_id: str,
    deployment_id: str,
    recovery_arguments: Sequence[str],
    client: CloudApiClient | None = None,
) -> None:
    """Stop following a Release Cloud won't publish.

    Cloud marks a held Release `superseded` when another one is selected while it waits
    for unfinished sessions. A Release that was never selected stays `smoke_tested`
    without a publication error, and Cloud refuses to retry it. One a promote selected
    before the hold stays `promoted`; with `client`, it is superseded only while the
    Agent has selected another Release, since a later rollback can select it again.
    """

    candidate = deployment.get("session_preflight")
    status = deployment.get("status")
    if (
        status not in _DEPLOYMENT_READY
        or not isinstance(candidate, dict)
        or candidate.get("state") != "superseded"
    ):
        return
    if status == "promoted":
        if client is None:
            return
        application = client.request("GET", f"/v1/applications/{application_id}")
        selected = application.get("current_deployment_id")
        if not isinstance(selected, str) or selected == deployment_id:
            return
    message = _deployment_failure_string(candidate.get("message"), max_bytes=4096)
    if message is None or safe_text(message, 4096) is None:
        message = (
            "Cayu Cloud selected another Release while this one waited for the serving "
            "release's unfinished sessions, so this one was not published."
        )
    raise _CloudReleaseSupersededError(
        message,
        application_id=application_id,
        deployment_id=deployment_id,
        recovery_arguments=recovery_arguments,
        status=str(status),
    )


def _cloud_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or len(value) > 64:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def _promote_release(
    client: CloudApiClient,
    *,
    application: dict[str, Any],
    deployment_id: str,
    wait: bool,
    poll_seconds: float,
    recovery_arguments: Sequence[str],
    wait_seconds: float,
    sleep: Callable[[float], None],
    monotonic: Callable[[], float],
    session_payload: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Promote a smoke-tested release, tolerating Cloud promoting it concurrently.

    Cloud's deployment worker promotes every smoke-tested release itself, reading the
    Agent's revision fresh. The CLI's promote carries the revision it read before the
    upload, so it loses that race whenever another release (often the previous one,
    still finishing) was promoted in between: Cloud answers 409 and then promotes this
    release anyway. A conflict is therefore only a failure when the release never
    becomes the promoted one.

    `session_payload` is a session choice the Release doesn't have yet. Cloud stores it
    once the promote selects the Release, so its publication uses it; a conflict leaves
    the Release's choice as it was.
    """

    application_id = str(application["id"])
    path = f"/v1/applications/{application_id}/deployments/{deployment_id}"
    try:
        promoted = client.request(
            "POST",
            f"{path}/promote",
            payload={
                "expected_application_revision": application["revision"],
                **(session_payload or {}),
            },
        )
    except CloudApiError as exc:
        if exc.status_code != 409:
            raise
        conflict = exc
    else:
        return promoted, client.request("GET", path)
    deadline = monotonic() + wait_seconds
    hold = _SessionHoldWatch(wait_seconds=wait_seconds, monotonic=monotonic)
    while True:
        deployment = client.request("GET", path)
        status = _deployment_status(deployment)
        if status == "promoted":
            return client.request("GET", f"/v1/applications/{application_id}"), deployment
        # Cloud won't promote a Release another one superseded while it waited.
        _raise_if_superseded(
            deployment,
            application_id=application_id,
            deployment_id=deployment_id,
            recovery_arguments=recovery_arguments,
        )
        publication_error = _deployment_publication_error(deployment)
        if status == "smoke_tested" and publication_error is not None:
            # Cloud refused to publish it (for example for unfinished sessions), so it
            # won't promote it on its own.
            raise _CloudServicePublicationError(
                publication_error,
                application_id=application_id,
                deployment_id=deployment_id,
                recovery_arguments=recovery_arguments,
                release=deployment,
            )
        if wait:
            deadline = hold.observe(deployment, deadline)
        if status in _DEPLOYMENT_FAILURES:
            _raise_terminal_deployment(
                application_id=application_id,
                deployment_id=deployment_id,
                status=status,
                recovery_arguments=recovery_arguments,
                failure=_deployment_failure(client, path=path),
            )
        if status != "smoke_tested" or not wait or monotonic() >= deadline:
            raise _CloudPromotionConflictError(
                conflict,
                application_id=application_id,
                deployment_id=deployment_id,
                recovery_arguments=recovery_arguments,
                status=status,
            )
        sleep(poll_seconds)


def _raise_terminal_deployment(
    *,
    application_id: str,
    deployment_id: str,
    status: str,
    recovery_arguments: Sequence[str],
    failure: _CloudDeploymentFailure | None,
    replayed: bool = False,
    retry_rejection: str | None = None,
) -> Never:
    details: dict[str, object] = {"status": status}
    app = _cloud_application_id(application_id)
    deployment = _cloud_deployment_id(deployment_id)
    if app is not None:
        details["application"] = app
    if deployment is not None:
        details["deployment_id"] = deployment
    if replayed:
        details["replayed"] = True
    if app is not None and deployment is not None:
        details["commands"] = {
            action: shlex.join(
                [
                    "cayu",
                    "cloud",
                    *recovery_arguments,
                    "deployment",
                    action,
                    deployment,
                    "--application",
                    app,
                ]
            )
            for action in (
                ("logs", "timeline", "retry")
                if retry_rejection is None and (failure is None or failure["automatic_retryable"])
                else ("logs", "timeline")
            )
        }
    if failure is not None and retry_rejection is not None:
        raise _CloudDeploymentFailureError(
            "deployment_retry_rejected", retry_rejection, failure=failure, details=details
        )
    if failure is not None:
        hint = (
            " Change the source and deploy again."
            if replayed and not failure["automatic_retryable"]
            else ""
        )
        raise _CloudDeploymentFailureError(
            "deployment_failed" if replayed else failure["code"],
            failure["message"] + hint,
            failure=failure,
            details=details,
        )
    details["diagnostic_status"] = "unavailable_or_unsupported"
    raise _CloudDeploymentDiagnosticUnavailableError(details)


def _deployment_failure(
    client: CloudApiClient,
    *,
    path: str,
) -> _CloudDeploymentFailure | None:
    try:
        timeline = client.request("GET", f"{path}/timeline")
    except CloudApiError:
        return None
    candidate = timeline.get("failure")
    if not isinstance(candidate, dict):
        return None
    if "schema_version" in candidate:
        parsed = parse_build_failure(candidate)
        if parsed is not None and _application_health_failure(parsed["code"], parsed["detail"]):
            parsed["automatic_retryable"] = False
        return parsed
    code = _deployment_failure_string(candidate.get("code"), max_bytes=64)
    phase = _deployment_failure_string(candidate.get("phase"), max_bytes=64)
    detail = _deployment_failure_string(candidate.get("detail"), max_bytes=4096)
    hint = _deployment_failure_string(candidate.get("hint"), max_bytes=1024)
    message = _deployment_failure_string(candidate.get("message"), max_bytes=512)
    if code is None or phase is None or detail is None or hint is None or message is None:
        return None
    automatic_retryable = candidate.get("automatic_retryable")
    if (
        code == "source_build_failed"
        and "schema_version" not in candidate
        and detail
        not in {
            "The Agent dependency set could not be resolved.",
            "The Agent Docker build context is invalid.",
            "The Agent package discovery configuration is ambiguous.",
            "The Agent Python project or lockfile is incomplete.",
            "The Agent source or a required package failed to compile.",
        }
    ):
        return None
    if (
        type(automatic_retryable) is not bool
        or re.fullmatch(r"[a-z][a-z0-9_]{0,63}", code) is None
        or re.fullmatch(r"[a-z][a-z0-9_]{0,63}", phase) is None
        or any(
            safe_text(candidate.get(key), limit) is None
            for key, limit in (("message", 512), ("detail", 4096), ("hint", 1024))
        )
    ):
        return None
    if _application_health_failure(code, detail):
        automatic_retryable = False
    return {
        "automatic_retryable": automatic_retryable,
        "code": code,
        "detail": detail,
        "hint": hint,
        "message": message,
        "phase": phase,
    }


def _application_health_failure(code: str, detail: str) -> bool:
    """Whether a smoke failure was the Agent's own health probe, on any smoke provider.

    Cloud reports it as "application process health check failed inside the release
    sandbox"; the E2B and Lambda MicroVM providers' raw wording ("... probe failed
    inside E2B" / "... inside the MicroVM") carries the same phrase.
    """

    return code == "release_smoke_failed" and "application process health" in detail.lower()


_RETRY_FAMILY_VERSION = re.compile(r"(.+?)(?:-retry-[0-9]+)*")


def _retry_family_root(version: str) -> str:
    """Version shared by an original Release and its retries, including nested ones."""

    match = _RETRY_FAMILY_VERSION.fullmatch(version)
    return (version if match is None else match.group(1))[:110]


def _latest_source_retry(
    client: CloudApiClient, application_id: str, original: dict[str, Any]
) -> dict[str, Any]:
    """Follow the immutable retry family before deciding whether to submit work.

    Members are the original's direct `-retry-N` children and, for deployments made
    by earlier clients and servers, nested `-retry-N-retry-M` versions.
    """

    version = original.get("version")
    digest = original.get("manifest_digest")
    if not isinstance(version, str) or not isinstance(digest, str):
        return original
    root = _retry_family_root(version)
    cursor = None
    seen = set()
    latest = original
    for _page in range(100):
        query = {"limit": "100"}
        if cursor is not None:
            query["cursor"] = cursor
        page = client.request("GET", f"/v1/applications/{application_id}/deployments", query=query)
        items = page.get("items")
        if not isinstance(items, list):
            raise CloudApiError(
                "api_response_invalid", "Cayu Cloud returned an invalid deployment list."
            )
        for item in items:
            if not isinstance(item, dict):
                continue
            candidate_version = item.get("version", "")
            if (
                isinstance(candidate_version, str)
                and item.get("id") != original.get("id")
                and _retry_family_root(candidate_version) == root
                and item.get("manifest_digest") == digest
                and item.get("policy_version") == original.get("policy_version")
                and isinstance(item.get("created_at"), str)
                and (
                    latest is original
                    or (item["created_at"], str(item.get("id")))
                    > (latest["created_at"], str(latest.get("id")))
                )
            ):
                _deployment_status(item)
                latest = item
        cursor = page.get("next_cursor")
        if cursor is None:
            return latest
        if not isinstance(cursor, str) or cursor in seen:
            break
        seen.add(cursor)
    raise CloudApiError(
        "api_response_invalid", "Cayu Cloud retry history pagination did not converge."
    )


def _deployment_failure_string(value: object, *, max_bytes: int) -> str | None:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or not value.isprintable()
        or len(value.encode("utf-8")) > max_bytes
    ):
        return None
    return value


def _deployment_status(deployment: dict[str, Any]) -> str:
    status = deployment.get("status")
    if not isinstance(status, str) or status not in _DEPLOYMENT_STATUSES:
        raise CloudApiError(
            "api_response_invalid",
            "Cayu Cloud returned an invalid deployment status.",
        )
    return status


def _read_or_none(client: CloudApiClient, path: str) -> dict[str, Any] | None:
    """GET a resource that may not exist yet, such as a pending service or an archive."""

    try:
        return client.request("GET", path)
    except CloudApiError as exc:
        if exc.status_code == 404:
            return None
        raise


def _deployment_publication_error(deployment: dict[str, Any]) -> dict[str, Any] | None:
    candidate = deployment.get("publication_error")
    if not isinstance(candidate, dict):
        return None
    error: dict[str, Any] = {}
    for key, max_bytes in (("code", 64), ("message", 512), ("detail", 4096), ("hint", 1024)):
        value = _deployment_failure_string(candidate.get(key), max_bytes=max_bytes)
        if value is not None:
            error[key] = value
    if not {"code", "message", "hint"} <= set(error):
        return None
    sessions = _publication_session_acknowledgements(candidate.get("acknowledge_sessions"))
    if sessions:
        error["acknowledge_sessions"] = sessions
    return error


def _publication_session_acknowledgements(value: object) -> list[str]:
    """The sessions a refusal asks a `proceed` retry to name; empty unless all are valid."""

    if not isinstance(value, list) or len(value) > _MAX_SESSION_ACKNOWLEDGEMENTS:
        return []
    sessions: set[str] = set()
    for item in value:
        if not isinstance(item, str) or not _valid_session_id(item):
            return []
        sessions.add(item)
    if _ACKNOWLEDGE_ALL_SESSIONS in sessions:
        return [_ACKNOWLEDGE_ALL_SESSIONS]
    return sorted(sessions)


def _transient_api_error(exc: CloudApiError) -> bool:
    return exc.category == "api_unavailable" or exc.status_code in _TRANSIENT_HTTP_STATUSES


def _web_not_ready_message(service: dict[str, Any]) -> str | None:
    for issue in _service_issues(service):
        if issue.get("code") == "web_not_ready":
            return _deployment_failure_string(issue.get("message"), max_bytes=512)
    return None


def _wait_for_service(
    client: CloudApiClient,
    *,
    application_id: str,
    initial: dict[str, Any] | None,
    expected_deployment_id: str | None = None,
    poll_seconds: float,
    recovery_arguments: Sequence[str] = (),
    wait_seconds: float,
    sleep: Callable[[float], None],
    monotonic: Callable[[], float],
) -> dict[str, Any]:
    _validate_wait(poll_seconds, wait_seconds)
    started = monotonic()
    deadline = started + wait_seconds
    path = f"/v1/applications/{application_id}/service"
    service = initial
    transient_failures = 0
    last_issue: str | None = None
    hold = _SessionHoldWatch(wait_seconds=wait_seconds, monotonic=monotonic)
    while True:
        pending = service is None or (
            expected_deployment_id is not None
            and service.get("deployment_id") != expected_deployment_id
        )
        if service is not None and service.get("status") == "archived":
            # Archive retires the service for good, even mid-deploy: never wait it out.
            raise archived_agent_error()
        status = "starting" if pending or service is None else _service_status(service)
        if pending and expected_deployment_id is not None:
            # A publication failure after promotion leaves the old or no service, and so
            # does a publication Cloud holds for the serving release's sessions.
            try:
                expected = client.request(
                    "GET",
                    f"/v1/applications/{application_id}/deployments/{expected_deployment_id}",
                )
            except CloudApiError as exc:
                if not _transient_api_error(exc):
                    raise
                expected = None
            publication_error = None
            if expected is not None:
                _raise_if_superseded(
                    expected,
                    application_id=application_id,
                    deployment_id=expected_deployment_id,
                    recovery_arguments=recovery_arguments,
                    client=client,
                )
                publication_error = _deployment_publication_error(expected)
                deadline = hold.observe(expected, deadline)
                last_issue = hold.waiting or last_issue
            if publication_error is not None:
                raise _CloudServicePublicationError(
                    publication_error,
                    application_id=application_id,
                    deployment_id=expected_deployment_id,
                    recovery_arguments=recovery_arguments,
                    release=expected,
                )
        elif service is not None:
            last_issue = _web_not_ready_message(service) or last_issue
        if status in _SERVICE_READY:
            assert service is not None
            return service
        if status in _SERVICE_FAILURES:
            assert service is not None
            issues = _service_issues(service)
            raise _CloudServiceHealthError(
                "service_degraded" if status == "degraded" else "service_failed",
                _service_failure_message(issues, status=str(status)),
                issues=issues,
            )
        now = monotonic()
        if now >= deadline:
            raise _CloudServiceStillRunningError(
                application_id=application_id,
                deleting=False,
                recovery_arguments=recovery_arguments,
                status=status,
                waited_seconds=now - started,
                last_issue=last_issue,
            )
        sleep(poll_seconds)
        try:
            service = _read_or_none(client, path)
        except CloudApiError as exc:
            # A brief API outage while the service starts is not a failed deploy.
            if not _transient_api_error(exc) or transient_failures >= _SERVICE_POLL_TRANSIENT_LIMIT:
                raise
            transient_failures += 1
            continue
        transient_failures = 0


def _service_status(service: dict[str, Any]) -> str:
    status = service.get("status")
    if not isinstance(status, str) or status not in _SERVICE_STATUSES:
        raise CloudApiError(
            "api_response_invalid",
            "Cayu Cloud returned an invalid Agent service status.",
        )
    return status


def _service_issues(service: dict[str, Any]) -> list[dict[str, Any]]:
    issues = service.get("issues")
    if not isinstance(issues, list):
        return []
    return [dict(issue) for issue in issues if isinstance(issue, dict)]


def _service_failure_message(issues: Sequence[dict[str, Any]], *, status: str) -> str:
    diagnostics: list[str] = []
    for issue in issues:
        message = issue.get("message")
        hint = issue.get("hint")
        parts = [value for value in (message, hint) if isinstance(value, str) and value]
        if parts:
            diagnostics.append(" ".join(parts))
    if diagnostics:
        return " ".join(diagnostics)
    return f"Agent service reached terminal status: {status}"


def _read_context(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise CloudCommandError("context_invalid", "Could not read Cayu Cloud context.") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or payload.get("status") != "ready"
        or not isinstance(payload.get("api_url"), str)
        or not payload["api_url"]
        or not isinstance(payload.get("api_key_file"), str)
        or not payload["api_key_file"]
    ):
        raise CloudCommandError("context_invalid", "Cayu Cloud context is invalid.")
    return payload


def _read_api_key_file(path: Path) -> str:
    try:
        api_key = path.expanduser().read_text().strip()
    except OSError as exc:
        raise CloudCommandError(
            "api_key_unavailable",
            "Could not read the Cayu Cloud API key file.",
        ) from exc
    if not api_key or any(character.isspace() for character in api_key):
        raise CloudCommandError(
            "api_key_unavailable",
            "Cayu Cloud API key must be a non-empty canonical value.",
        )
    return api_key


def _context_file_path(context_path: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return context_path.expanduser().resolve().parent / path


def _selected_context_path(explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit.expanduser()
    configured = os.environ.get("CAYU_CLOUD_CONTEXT")
    if configured:
        return Path(configured).expanduser()
    config_path = _active_config_path()
    try:
        payload = json.loads(config_path.read_text())
    except FileNotFoundError:
        return None
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise CloudCommandError(
            "context_not_selected",
            "No valid Cayu Cloud context is selected.",
        ) from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or set(payload) != {"active_context", "schema_version"}
        or not isinstance(payload.get("active_context"), str)
    ):
        raise CloudCommandError(
            "context_not_selected",
            "No valid Cayu Cloud context is selected.",
        )
    selected = Path(payload["active_context"])
    if not selected.is_absolute():
        raise CloudCommandError(
            "context_not_selected",
            "The active Cayu Cloud context reference is not absolute.",
        )
    return selected


def _selected_context(explicit: Path | None) -> tuple[Path, dict[str, Any]]:
    selected = _selected_context_path(explicit)
    if selected is None:
        raise CloudCommandError(
            "context_not_selected",
            "Select a Cayu Cloud context with `cayu cloud context use PATH`.",
        )
    return selected, _read_context(selected)


def _context_source(explicit: Path | None) -> str:
    if explicit is not None:
        return "explicit"
    if os.environ.get("CAYU_CLOUD_CONTEXT"):
        return "environment"
    return "persisted"


def _configured_cloud_api_url(
    arguments: argparse.Namespace,
    *,
    context: dict[str, Any] | None,
    context_path: Path | None,
) -> str | None:
    def context_url() -> str | None:
        if context_path is None:
            return None
        selected_context = context if context is not None else _read_context(context_path)
        return str(selected_context["api_url"]).rstrip("/")

    if context_path is not None and _context_source(arguments.context) != "persisted":
        return context_url()
    environment_url = os.environ.get("CAYU_CLOUD_API_URL")
    if environment_url:
        return environment_url.rstrip("/")
    return context_url()


def _cloud_login_authority_fingerprint(credentials: CloudAuthCredentials) -> str:
    authority = {
        "api_url": credentials.api_url,
        "organization_id": credentials.organization_id,
        "user_id": credentials.user_id,
        "workos_api_hostname": credentials.workos_api_hostname,
        "workos_client_id": credentials.workos_client_id,
    }
    payload = json.dumps(authority, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(b"cayu.cloud-login-authority.v1\0" + payload).hexdigest()


def _cloud_client(
    arguments: argparse.Namespace,
    *,
    context: dict[str, Any] | None = None,
    context_path: Path | None = None,
) -> CloudApiClient:
    context = context or {}
    context_is_selected = context_path is not None
    configured_api_url = _configured_cloud_api_url(
        arguments,
        context=context,
        context_path=context_path,
    )
    environment_key_file = os.environ.get("CAYU_CLOUD_API_KEY_FILE") or None
    key_file_value = arguments.api_key_file or environment_key_file
    environment_key = os.environ.get("CAYU_CLOUD_API_KEY")
    if environment_key:
        api_url = configured_api_url or _PRODUCTION_API_URL
        return CloudApiClient(
            api_url=str(api_url).rstrip("/"),
            api_key=environment_key,
            timeout_seconds=arguments.timeout_seconds,
        )
    if context_is_selected and key_file_value is None:
        context_api_url = str(context["api_url"]).rstrip("/")
        if configured_api_url != context_api_url:
            raise CloudAuthError(
                "context_api_mismatch",
                "CAYU_CLOUD_API_URL differs from the persisted context; provide "
                "an explicit API key for that Cayu Cloud or select a matching context.",
            )
    if key_file_value is not None or context_is_selected:
        api_url = configured_api_url or _PRODUCTION_API_URL
        selected_key_file = key_file_value or context.get("api_key_file")
        if not selected_key_file:
            raise CloudApiError(
                "api_key_unavailable",
                "CAYU_CLOUD_API_KEY or a private API-key file is required.",
            )
        key_file_path = Path(selected_key_file).expanduser()
        if (
            arguments.api_key_file is None
            and environment_key_file is None
            and context_path is not None
        ):
            key_file_path = _context_file_path(context_path, str(selected_key_file))
        return CloudApiClient.from_key_file(
            api_url=str(api_url),
            api_key_file=key_file_path,
            timeout_seconds=arguments.timeout_seconds,
        )

    auth_store = CloudAuthStore()
    credentials = auth_store.load()
    if credentials is not None:
        api_url = _PRODUCTION_API_URL
        if api_url != credentials.api_url:
            raise CloudAuthError(
                "login_api_mismatch",
                "The saved login belongs to another Cayu Cloud; run "
                "`cayu cloud login` to sign in to production.",
            )
        authority_fingerprint = _cloud_login_authority_fingerprint(credentials)

        def require_same_login_authority(current: CloudAuthCredentials) -> None:
            if _cloud_login_authority_fingerprint(current) != authority_fingerprint:
                raise CloudAuthError(
                    "login_authority_changed",
                    "The Cayu Cloud login authority changed; run `cayu cloud login` again.",
                )

        def current_access_token(rejected: str | None = None) -> str:
            current = fresh_cloud_credentials(
                auth_store,
                timeout_seconds=arguments.timeout_seconds,
                rejected_access_token=rejected,
            )
            if current is None:
                raise CloudAuthError(
                    "cloud_auth_unavailable",
                    "Could not read the local Cayu Cloud login.",
                )
            require_same_login_authority(current)
            return current.access_token

        credentials = fresh_cloud_credentials(
            auth_store,
            timeout_seconds=arguments.timeout_seconds,
        )
        if credentials is None:
            raise CloudAuthError(
                "cloud_auth_unavailable",
                "Could not read the local Cayu Cloud login.",
            )
        require_same_login_authority(credentials)
        return CloudApiClient(
            api_url=api_url,
            api_key=credentials.access_token,
            timeout_seconds=arguments.timeout_seconds,
            api_key_provider=current_access_token,
            # A long deploy wait can outlive a token Cloud rejects before the local
            # clock says it expires; refresh it once instead of failing the wait.
            rejected_api_key_provider=current_access_token,
        )

    raise CloudAuthError(
        "login_required",
        "Sign in with `cayu cloud login`, select a private context, or provide an API key.",
    )


def _evidence_directory(arguments: argparse.Namespace) -> Path:
    if arguments.evidence_dir is not None:
        return arguments.evidence_dir.expanduser().resolve()
    configured = os.environ.get("CAYU_CLOUD_EVIDENCE_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path.home() / ".local" / "state" / "cayu-cloud" / "evidence"


def _cloud_recovery_arguments(arguments: argparse.Namespace) -> tuple[str, ...]:
    result: list[str] = []
    for flag, value in (
        ("--context", getattr(arguments, "context", None)),
        ("--api-key-file", getattr(arguments, "api_key_file", None)),
    ):
        if value is not None:
            result.extend((flag, str(value)))
    return tuple(result)


def _correlation_id(operation: str) -> str:
    return f"{operation}:{secrets.token_hex(12)}"


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _application_slug(value: str) -> str:
    if not is_application_slug(value):
        raise CloudCommandError(
            "invalid_input",
            "Application must be an 8-63 character lowercase Cayu Cloud slug "
            "containing only letters, numbers, and interior hyphens.",
        )
    return value


def _validate_wait(poll_seconds: float, wait_seconds: float) -> None:
    if (
        not math.isfinite(poll_seconds)
        or not math.isfinite(wait_seconds)
        or poll_seconds <= 0
        or wait_seconds <= 0
        or poll_seconds > wait_seconds
    ):
        raise CloudApiError(
            "invalid_wait",
            "poll-seconds and wait-seconds must be positive and ordered.",
        )


def _preflight_wait(arguments: argparse.Namespace) -> None:
    _validate_wait(arguments.poll_seconds, arguments.wait_seconds)


def _preflight_deploy_wait(arguments: argparse.Namespace) -> None:
    _acknowledged_revisions(arguments)
    _session_choice(arguments)
    if not arguments.no_wait:
        _validate_wait(arguments.poll_seconds, arguments.wait_seconds)


def _preflight_acknowledgements(arguments: argparse.Namespace) -> None:
    _acknowledged_revisions(arguments)
    _session_choice(arguments)


def _preflight_rollback_wait(arguments: argparse.Namespace) -> None:
    _session_choice(arguments)
    if arguments.wait:
        _validate_wait(arguments.poll_seconds, arguments.wait_seconds)


def _positive_finite_seconds(value: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a finite positive number") from exc
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return result


def _active_config_path() -> Path:
    configured = os.environ.get("CAYU_CLOUD_CONFIG")
    if configured:
        return Path(configured).expanduser().resolve()
    config_home = os.environ.get("XDG_CONFIG_HOME")
    root = Path(config_home).expanduser() if config_home else Path.home() / ".config"
    return root / "cayu-cloud" / "config.json"
