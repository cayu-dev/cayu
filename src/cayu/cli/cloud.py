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
from pathlib import Path
from typing import Any, Never, TypedDict

from cayu.cli._cloud_api import CloudApiClient, CloudApiError
from cayu.cli._cloud_auth import (
    CloudAuthCredentials,
    CloudAuthError,
    CloudAuthStore,
    WorkOSDeviceAuthClient,
    fresh_cloud_credentials,
)
from cayu.cli._cloud_evidence import EvidenceRecorder
from cayu.cli._cloud_private_state import write_private_json as _write_private_json
from cayu.cli._cloud_project import (
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
_SERVICE_STATUSES = _SERVICE_FAILURES | _SERVICE_IN_PROGRESS | _SERVICE_READY
_CLOUD_DEPLOYMENT_ID = re.compile(r"dep_[a-z0-9]{1,64}")
_PRODUCTION_API_URL = "https://cloud.cayu.dev"
_SOURCE_BUILD_FAILURE_MESSAGE = "The Agent image could not be built."
_SOURCE_BUILD_FAILURE_HINT = "Fix the Agent source and deploy again."
_SOURCE_BUILD_FAILURE_DETAILS = frozenset(
    {
        "The Agent dependency set could not be resolved.",
        "The Agent Docker build context is invalid.",
        "The Agent package discovery configuration is ambiguous.",
        "The Agent Python project or lockfile is incomplete.",
        "The Agent source or a required package failed to compile.",
    }
)


class _CloudDeploymentFailure(TypedDict):
    automatic_retryable: bool
    code: str
    detail: str
    hint: str
    message: str
    phase: str


class CloudCommandError(RuntimeError):
    """Stable customer-facing Cloud command failure."""

    def __init__(self, category: str, message: str) -> None:
        super().__init__(message)
        self.category = category


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


class _CloudDeploymentFailureError(CloudApiError):
    """Safe structured Cayu Cloud deployment failure."""

    def __init__(
        self,
        category: str,
        message: str,
        *,
        failure: _CloudDeploymentFailure,
    ) -> None:
        super().__init__(category, message)
        self.failure = failure.copy()


class _CloudDeploymentStillRunningError(CloudApiError):
    """A local wait ended while Cayu Cloud retained the deployment operation."""

    def __init__(
        self,
        *,
        application_id: str,
        deployment_id: str,
        recovery_arguments: Sequence[str] = (),
        status: str,
    ) -> None:
        super().__init__(
            "deployment_still_running",
            "Cayu Cloud is still processing this deployment.",
        )
        self.application_id = _cloud_application_id(application_id)
        self.deployment_id = _cloud_deployment_id(deployment_id)
        self.recovery_arguments = tuple(recovery_arguments)
        self.status = status if status in _DEPLOYMENT_IN_PROGRESS else "processing"

    def public_details(self) -> dict[str, object]:
        details: dict[str, object] = {"status": self.status}
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


class _CloudServiceStillRunningError(CloudApiError):
    """A local wait ended while Cayu Cloud retained a service operation."""

    def __init__(
        self,
        *,
        application_id: str,
        deleting: bool,
        recovery_arguments: Sequence[str] = (),
        status: str,
    ) -> None:
        super().__init__(
            "service_deletion_still_running" if deleting else "service_still_starting",
            (
                "Cayu Cloud is still deleting this Agent service."
                if deleting
                else "Cayu Cloud is still starting this Agent service."
            ),
        )
        self.application_id = _cloud_application_id(application_id)
        self.recovery_arguments = tuple(recovery_arguments)
        self.status = status if status in _SERVICE_IN_PROGRESS else "processing"

    def public_details(self) -> dict[str, object]:
        details: dict[str, object] = {"status": self.status}
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
    if isinstance(exc, _CloudDeploymentFailureError):
        error["failure"] = exc.failure
    if isinstance(exc, (_CloudDeploymentStillRunningError, _CloudServiceStillRunningError)):
        error.update(exc.public_details())
    if isinstance(exc, _CloudServiceHealthError):
        error["issues"] = exc.issues
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
    application_commands.add_parser(
        "list",
        description="List Agent applications and their selected releases.",
    )

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
        description="Generate cayu-cloud.toml from a Python Agent project.",
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
    }
    for action, description in deployment_descriptions.items():
        operation = deployment_commands.add_parser(action, description=description)
        operation.add_argument("deployment_id")
        operation.add_argument("--application", required=True)
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


def _execute(arguments: argparse.Namespace) -> dict[str, Any]:
    preflight: Callable[[argparse.Namespace], None] | None = arguments._cloud_preflight
    if preflight is not None:
        preflight(arguments)
    if arguments.command == "init":
        initialized = initialize_project(arguments.path, force=arguments.force)
        return {
            "operation": "init",
            "result": {
                "application": initialized.application,
                "manifest": str(initialized.manifest_path),
                "name": initialized.name,
                "runtime": initialized.runtime,
            },
        }
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
        return {
            "operation": "applications.list",
            "result": client.request("GET", "/v1/applications"),
        }
    if arguments.command == "deploy":
        return _deploy(
            arguments,
            client=client,
            recorder=recorder,
            project=project,
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


def _deploy(
    arguments: argparse.Namespace,
    *,
    client: CloudApiClient,
    recorder: EvidenceRecorder,
    project: ResolvedCloudProject,
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
    deployment = client.request(
        "POST",
        f"/v1/applications/{application['id']}/deployments",
        payload=project.manifest.deployment_payload(
            repository=repository,
            revision=revision,
        ),
        idempotency_key=_deployment_idempotency_key(
            application_id=str(application["id"]),
            version=project.manifest.version,
            revision=revision,
        ),
    )
    if not arguments.no_wait:
        deployment = _wait_for_deployment(
            client,
            application_id=str(application["id"]),
            deployment_id=str(deployment["id"]),
            include_failure_diagnostics=True,
            poll_seconds=arguments.poll_seconds,
            recovery_arguments=_cloud_recovery_arguments(arguments),
            wait_seconds=arguments.wait_seconds,
            sleep=sleep,
            monotonic=monotonic,
        )
    if not arguments.no_promote and deployment.get("status") == "smoke_tested":
        application = client.request(
            "POST",
            f"/v1/applications/{application['id']}/deployments/{deployment['id']}/promote",
            payload={"expected_application_revision": application["revision"]},
        )
        deployment = client.request(
            "GET",
            f"/v1/applications/{application['id']}/deployments/{deployment['id']}",
        )
    runtime_artifact = None
    runtime_artifact_id = deployment.get("runtime_artifact_id")
    if isinstance(runtime_artifact_id, str) and runtime_artifact_id:
        runtime_artifact = client.request(
            "GET",
            f"/v1/applications/{application['id']}/runtime-artifacts/{runtime_artifact_id}",
        )
    service = None
    if project.manifest.runtime_payload() is not None and deployment.get("status") == "promoted":
        service = client.request(
            "PUT",
            f"/v1/applications/{application['id']}/service",
        )
        if not arguments.no_wait:
            service = _wait_for_service(
                client,
                application_id=str(application["id"]),
                initial=service,
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
        "source": {
            "content_digest": project.content_digest,
            "kind": "local_bundle" if project.bundle is not None else "repository",
            "manifest": str(project.manifest_path),
            "repository": repository,
            "revision": revision,
        },
    }
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


def _deployment_idempotency_key(
    *,
    application_id: str,
    version: str,
    revision: str,
) -> str:
    identity = json.dumps(
        [application_id, version, revision],
        ensure_ascii=True,
        separators=(",", ":"),
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
        result = client.request("GET", f"{base}/{arguments.deployment_command}")
    elif arguments.deployment_command == "wait":
        result = _wait_for_deployment(
            client,
            application_id=application_id,
            deployment_id=arguments.deployment_id,
            poll_seconds=arguments.poll_seconds,
            recovery_arguments=_cloud_recovery_arguments(arguments),
            wait_seconds=arguments.wait_seconds,
            sleep=sleep,
            monotonic=monotonic,
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
) -> dict[str, Any]:
    application_id = _application_id(client, arguments.application)
    application = client.request("GET", f"/v1/applications/{application_id}")
    selected = client.request(
        "POST",
        (f"/v1/applications/{application_id}/deployments/{arguments.deployment_id}/rollback"),
        payload={"expected_application_revision": application["revision"]},
    )
    service = client.request(
        "PUT",
        f"/v1/applications/{application_id}/service",
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


def _application_id(client: CloudApiClient, reference: str) -> str:
    applications = client.request("GET", "/v1/applications").get("items", [])
    if not isinstance(applications, list):
        raise CloudApiError("api_response_invalid", "Application list is invalid.")
    matches = [
        application
        for application in applications
        if isinstance(application, dict)
        and (
            application.get("id") == reference
            or application.get("name") == reference
            or _slug(str(application.get("name", ""))) == reference
        )
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
) -> dict[str, Any]:
    _validate_wait(poll_seconds, wait_seconds)
    deadline = monotonic() + wait_seconds
    path = f"/v1/applications/{application_id}/deployments/{deployment_id}"
    while True:
        deployment = client.request("GET", path)
        status = _deployment_status(deployment)
        if status in _DEPLOYMENT_READY:
            return deployment
        if status in _DEPLOYMENT_FAILURES:
            if status == "failed" and include_failure_diagnostics:
                failure = _deployment_failure(client, path=path)
                if failure is not None:
                    raise _CloudDeploymentFailureError(
                        failure["code"],
                        failure["message"],
                        failure=failure,
                    )
            raise CloudApiError(
                "deployment_failed",
                f"Deployment reached terminal status: {status}",
            )
        if monotonic() >= deadline:
            raise _CloudDeploymentStillRunningError(
                application_id=application_id,
                deployment_id=deployment_id,
                recovery_arguments=recovery_arguments,
                status=status,
            )
        sleep(poll_seconds)


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
    code = _deployment_failure_string(candidate.get("code"), max_bytes=64)
    phase = _deployment_failure_string(candidate.get("phase"), max_bytes=64)
    detail = _deployment_failure_string(candidate.get("detail"), max_bytes=4096)
    hint = _deployment_failure_string(candidate.get("hint"), max_bytes=1024)
    message = _deployment_failure_string(candidate.get("message"), max_bytes=512)
    if code is None or phase is None or detail is None or hint is None or message is None:
        return None
    automatic_retryable = candidate.get("automatic_retryable")
    if (
        automatic_retryable is not False
        or code != "source_build_failed"
        or phase != "image_built"
        or message != _SOURCE_BUILD_FAILURE_MESSAGE
        or hint != _SOURCE_BUILD_FAILURE_HINT
        or detail not in _SOURCE_BUILD_FAILURE_DETAILS
    ):
        return None
    return {
        "automatic_retryable": automatic_retryable,
        "code": code,
        "detail": detail,
        "hint": hint,
        "message": message,
        "phase": phase,
    }


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


def _wait_for_service(
    client: CloudApiClient,
    *,
    application_id: str,
    initial: dict[str, Any],
    poll_seconds: float,
    recovery_arguments: Sequence[str] = (),
    wait_seconds: float,
    sleep: Callable[[float], None],
    monotonic: Callable[[], float],
) -> dict[str, Any]:
    _validate_wait(poll_seconds, wait_seconds)
    deadline = monotonic() + wait_seconds
    path = f"/v1/applications/{application_id}/service"
    service = initial
    while True:
        status = _service_status(service)
        if status in _SERVICE_READY:
            return service
        if status in _SERVICE_FAILURES:
            issues = _service_issues(service)
            raise _CloudServiceHealthError(
                "service_degraded" if status == "degraded" else "service_failed",
                _service_failure_message(issues, status=str(status)),
                issues=issues,
            )
        if monotonic() >= deadline:
            raise _CloudServiceStillRunningError(
                application_id=application_id,
                deleting=False,
                recovery_arguments=recovery_arguments,
                status=status,
            )
        sleep(poll_seconds)
        service = client.request("GET", path)


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

        def current_access_token() -> str:
            current = fresh_cloud_credentials(
                auth_store,
                timeout_seconds=arguments.timeout_seconds,
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
    if not arguments.no_wait:
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
