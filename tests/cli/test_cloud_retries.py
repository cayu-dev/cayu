from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from cayu.cli import cloud
from cayu.cli._cloud_api import CloudApiError, _plain_api_error_detail
from cayu.cli._cloud_evidence import EvidenceRecorder
from cayu.cli._cloud_project import CloudProjectManifest, ResolvedCloudProject


def _failure(*, retryable=True):
    return {
        "schema_version": 1,
        "code": "release_smoke_failed" if retryable else "source_build_inputs_invalid",
        "phase": "smoke_tested" if retryable else "source_resolved",
        "message": "The release smoke test failed."
        if retryable
        else "The Agent build inputs are incomplete.",
        "detail": "The release smoke test did not complete successfully."
        if retryable
        else "The source has no lockfile.",
        "hint": "Inspect the deployment diagnostics."
        if retryable
        else "Change the source and deploy again.",
        "automatic_retryable": retryable,
        "attempt": 8,
        "diagnostic_ref": None,
        "diagnostic": {
            "status": "unavailable",
            "stage": "image_build",
            "reason": "not_recorded",
            "exit_code": None,
            "excerpt": "",
            "truncated": False,
        },
    }


def _project(tmp_path):
    manifest = CloudProjectManifest.loads("""
schema_version = 1
application = "retry-agent"
name = "Retry Agent"
version = "1.0.0"
entrypoint = "python -m agent"
capabilities = ["model.generate"]
cpu_millis = 512
memory_mb = 1024
timeout_seconds = 600
environment = "python"
compatibility = "cayu>=0.1"
policy_version = "v1"
""")
    bundle = b"unchanged-agent-archive"
    digest = hashlib.sha256(bundle).hexdigest()
    return ResolvedCloudProject(
        root=tmp_path,
        manifest_path=tmp_path / "cayu-cloud.toml",
        manifest=manifest,
        repository="cayu-cloud://source-bundles/sha256/" + digest,
        revision=digest[:40],
        bundle=bundle,
        content_digest="sha256:" + digest,
    )


class Client:
    def __init__(self, project, *, status="destroyed", failure=None):
        self.project = project
        self.status = status
        self.failure = _failure() if failure is None else failure
        self.requests = []
        self.uploads = []

    def upload_bytes(self, url, bundle):
        self.uploads.append(bundle)

    def request(self, method, path, **kwargs):
        self.requests.append((method, path, kwargs))
        if path == "/v1/applications":
            return {"items": [{"id": "retry-agent", "name": "Retry Agent", "revision": 1}]}
        if path == "/v1/source-bundles/uploads":
            return {
                "content_digest": self.project.content_digest,
                "repository": self.project.repository,
                "revision": self.project.revision,
                "size_bytes": len(self.project.bundle),
                "upload_url": "https://uploads.example.test/signed",
            }
        if method == "POST" and path.endswith("/deployments"):
            return {"id": "dep_old", "status": self.status}
        if path.endswith("/timeline"):
            return {"failure": self.failure}
        if method == "POST" and path.endswith("/retry"):
            return {"id": "dep_new", "status": "accepted"}
        raise AssertionError((method, path))


def _deploy(client, project, tmp_path, *, retry_failed=True):
    return cloud._deploy(
        SimpleNamespace(application=None, no_wait=True, no_promote=True, retry_failed=retry_failed),
        client=client,
        project=project,
        recorder=EvidenceRecorder(tmp_path / "evidence"),
    )


@pytest.mark.parametrize("status", ["failed", "destroyed"])
def test_unchanged_bundle_retries_cloud_failure_with_one_new_submission_key(tmp_path: Path, status):
    project = _project(tmp_path)
    client = Client(project, status=status)
    first = _deploy(client, project, tmp_path)
    second = _deploy(client, project, tmp_path)
    assert first["result"]["retry"] == {
        "previous_deployment_id": "dep_old",
        "deployment_id": "dep_new",
        "failure_code": "release_smoke_failed",
        "reason": "The release smoke test failed.",
    }
    assert second["result"]["deployment"]["id"] == "dep_new"
    creates = [
        kwargs
        for method, path, kwargs in client.requests
        if method == "POST" and path.endswith("/deployments")
    ]
    retries = [
        kwargs
        for method, path, kwargs in client.requests
        if method == "POST" and path.endswith("/retry")
    ]
    assert creates[0] == creates[1]
    assert retries[0]["idempotency_key"] == retries[1]["idempotency_key"]
    assert client.uploads == [project.bundle, project.bundle]


@pytest.mark.parametrize("status", ["accepted", "promoted"])
def test_inflight_and_successful_replay_keep_the_same_deployment(tmp_path: Path, status):
    project = _project(tmp_path)
    client = Client(project, status=status)
    result = _deploy(client, project, tmp_path)
    assert result["result"]["deployment"]["id"] == "dep_old"
    assert "retry" not in result["result"]
    assert not any(path.endswith(("/timeline", "/retry")) for _, path, _ in client.requests)


@pytest.mark.parametrize("retryable,retry_failed", [(False, True), (True, False)])
def test_replayed_terminal_failure_has_diagnostics_and_creates_nothing(
    tmp_path: Path, retryable, retry_failed
):
    project = _project(tmp_path)
    client = Client(project, failure=_failure(retryable=retryable))
    with pytest.raises(CloudApiError) as raised:
        _deploy(client, project, tmp_path, retry_failed=retry_failed)
    assert raised.value.details["deployment_id"] == "dep_old"
    assert raised.value.details["status"] == "destroyed"
    assert raised.value.details["replayed"] is True
    assert raised.value.failure["code"] == client.failure["code"]
    assert "timeline dep_old" in raised.value.details["commands"]["timeline"]
    if retryable:
        assert "retry dep_old" in raised.value.details["commands"]["retry"]
    else:
        assert "retry" not in raised.value.details["commands"]
    if not retryable:
        assert "Change the source and deploy again" in str(raised.value)
    assert not any(path.endswith("/retry") for _, path, _ in client.requests)


def test_public_retry_command_preserves_explicit_submission_key(tmp_path: Path):
    project = _project(tmp_path)
    client = Client(project)
    arguments = cloud._build_parser().parse_args(
        [
            "deployment",
            "retry",
            "dep_old",
            "--application",
            "retry-agent",
            "--idempotency-key",
            "one-submission",
        ]
    )
    result = cloud._deployment(arguments, client=client)
    assert result["result"]["id"] == "dep_new"
    assert client.requests[-1] == (
        "POST",
        "/v1/applications/retry-agent/deployments/dep_old/retry",
        {"idempotency_key": "one-submission"},
    )


def test_retry_flag_can_be_disabled_explicitly():
    arguments = cloud._build_parser().parse_args(["deploy", ".", "--no-retry-failed"])
    assert arguments.retry_failed is False


@pytest.mark.parametrize("status", ["accepted", "promoted", "failed"])
def test_deploy_follows_latest_retry_of_unchanged_source(tmp_path, status):
    project = _project(tmp_path)

    class HistoryClient(Client):
        def request(self, method, path, **kwargs):
            if method == "POST" and path.endswith("/deployments"):
                self.requests.append((method, path, kwargs))
                return {
                    "id": "dep_old",
                    "status": "destroyed",
                    "version": "1.0.0",
                    "manifest_digest": "sha256:same",
                    "policy_version": "v1",
                }
            if method == "GET" and "deployments?" in path:
                return {
                    "items": [
                        {
                            "id": "dep_child",
                            "version": "1.0.0-retry-1",
                            "manifest_digest": "sha256:same",
                            "policy_version": "v1",
                            "created_at": "2026-09-30T00:00:00Z",
                            "status": status,
                        }
                    ],
                    "next_cursor": None,
                }
            return super().request(method, path, **kwargs)

    client = HistoryClient(project)
    result = _deploy(client, project, tmp_path)
    retries = [(path, args) for method, path, args in client.requests if path.endswith("/retry")]
    if status == "failed":
        # The original is retried; the key names the newest failed attempt.
        assert len(retries) == 1 and retries[0][0].endswith("/dep_old/retry")
        assert retries[0][1]["idempotency_key"] == _retry_key(client, "dep_child")
        assert result["result"]["retry"]["previous_deployment_id"] == "dep_child"
    else:
        assert result["result"]["deployment"]["id"] == "dep_child"
        assert not retries


def _retry_key(client, failed_id):
    create = next(
        kwargs
        for method, path, kwargs in client.requests
        if method == "POST" and path.endswith("/deployments")
    )
    source_key = create["idempotency_key"]
    return "deploy-retry:" + hashlib.sha256(f"{source_key}:{failed_id}".encode()).hexdigest()


class FamilyClient(Client):
    """A failed original whose family has direct, nested, and unrelated versions."""

    def __init__(self, project, items):
        super().__init__(project, status="failed")
        self.items = items

    def request(self, method, path, **kwargs):
        if method == "POST" and path.endswith("/deployments"):
            self.requests.append((method, path, kwargs))
            return {
                "id": "dep_old",
                "status": "failed",
                "version": "1.0.0",
                "manifest_digest": "sha256:same",
                "policy_version": "v1",
                "created_at": "2026-09-30T00:00:00Z",
            }
        if method == "GET" and "deployments?" in path:
            self.requests.append((method, path, kwargs))
            return {"items": self.items, "next_cursor": None}
        return super().request(method, path, **kwargs)


def _member(deployment_id, version, created_at, status="failed", digest="sha256:same"):
    return {
        "id": deployment_id,
        "version": version,
        "manifest_digest": digest,
        "policy_version": "v1",
        "created_at": created_at,
        "status": status,
    }


def test_deploy_retries_the_original_after_nested_attempts_from_older_clients(tmp_path):
    project = _project(tmp_path)
    client = FamilyClient(
        project,
        [
            _member("dep_old", "1.0.0", "2026-09-30T00:00:00Z"),
            _member("dep_one", "1.0.0-retry-1", "2026-09-30T01:00:00Z"),
            _member("dep_nested", "1.0.0-retry-1-retry-1", "2026-09-30T02:00:00Z"),
            # Same version prefix, different source: not part of this family.
            _member("dep_other", "1.0.0-retry-2", "2026-09-30T03:00:00Z", digest="sha256:x"),
            _member("dep_unrelated", "1.0.0-retry-x", "2026-09-30T04:00:00Z"),
        ],
    )
    result = _deploy(client, project, tmp_path)
    retries = [(path, args) for method, path, args in client.requests if path.endswith("/retry")]
    assert [path for path, _ in retries] == [
        "/v1/applications/retry-agent/deployments/dep_old/retry"
    ]
    assert retries[0][1]["idempotency_key"] == _retry_key(client, "dep_nested")
    assert result["result"]["retry"]["previous_deployment_id"] == "dep_nested"


def test_deploy_waits_on_a_live_nested_attempt_instead_of_retrying(tmp_path):
    project = _project(tmp_path)
    client = FamilyClient(
        project,
        [
            _member("dep_one", "1.0.0-retry-1", "2026-09-30T01:00:00Z"),
            _member("dep_nested", "1.0.0-retry-1-retry-1", "2026-09-30T02:00:00Z", "accepted"),
        ],
    )
    result = _deploy(client, project, tmp_path)
    assert result["result"]["deployment"]["id"] == "dep_nested"
    assert not any(path.endswith("/retry") for _, path, _ in client.requests)


class RejectingClient(Client):
    def __init__(self, project, detail):
        super().__init__(project)
        self.detail = detail

    def request(self, method, path, **kwargs):
        if path.endswith("/retry"):
            self.requests.append((method, path, kwargs))
            raise CloudApiError(
                "api_request_rejected",
                "Cayu Cloud API returned HTTP 409.",
                status_code=409,
                detail=self.detail,
            )
        return super().request(method, path, **kwargs)


def test_legacy_server_rejection_retains_original_failure(tmp_path):
    project = _project(tmp_path)
    client = RejectingClient(project, "Only paused or failed deployments can be retried.")
    with pytest.raises(CloudApiError) as raised:
        _deploy(client, project, tmp_path)
    assert raised.value.category == "deployment_failed"
    assert raised.value.details["deployment_id"] == "dep_old"
    assert raised.value.failure["code"] == "release_smoke_failed"
    assert "retry" in raised.value.details["commands"]


@pytest.mark.parametrize(
    "detail",
    [
        "Retained source bundle is no longer available. Upload the source and deploy again.",
        "Concurrent retries changed the deployment family. Read its latest attempt and retry.",
        None,
    ],
)
def test_other_retry_conflicts_surface_the_server_reason_without_a_retry_command(
    tmp_path, capsys, detail
):
    project = _project(tmp_path)
    client = RejectingClient(project, detail)
    with pytest.raises(CloudApiError) as raised:
        _deploy(client, project, tmp_path)
    assert cloud._cloud_failure(raised.value) == 2
    error = json.loads(capsys.readouterr().out)["error"]
    assert error["category"] == "deployment_retry_rejected"
    assert error["message"] == (detail or "Cayu Cloud API returned HTTP 409.")
    assert error["deployment_id"] == "dep_old"
    assert error["failure"]["code"] == "release_smoke_failed"
    assert set(error["commands"]) == {"logs", "timeline"}


def test_public_retry_command_reports_the_server_conflict_reason():
    class Conflict:
        def request(self, method, path, **kwargs):
            if path == "/v1/applications":
                return {"items": [{"id": "retry-agent", "name": "Retry Agent", "revision": 1}]}
            raise CloudApiError(
                "api_request_rejected",
                "Cayu Cloud API returned HTTP 409.",
                status_code=409,
                detail="Retained source bundle is no longer available. Upload the source and deploy again.",
            )

    arguments = cloud._build_parser().parse_args(
        ["deployment", "retry", "dep_old", "--application", "retry-agent"]
    )
    with pytest.raises(CloudApiError) as raised:
        cloud._deployment(arguments, client=Conflict())
    assert str(raised.value) == (
        "Cayu Cloud API returned HTTP 409: Retained source bundle is no longer available."
        " Upload the source and deploy again."
    )


@pytest.mark.parametrize(
    "payload,expected",
    [
        (
            {"detail": "Only paused or failed deployments can be retried."},
            "Only paused or failed deployments can be retried.",
        ),
        ({"detail": "token: leaked-value"}, None),
        ({"detail": {"code": "agent_slug_invalid"}}, None),
        ({"detail": "x" * 513}, None),
        (["detail"], None),
    ],
)
def test_api_error_keeps_only_a_safe_plain_text_detail(payload, expected):
    response = httpx.Response(409, json=payload)
    assert _plain_api_error_detail(response) == expected


def test_interrupted_retry_exposes_stable_reconciliation_key(tmp_path):
    project = _project(tmp_path)

    class UnavailableClient(Client):
        def request(self, method, path, **kwargs):
            if path.endswith("/retry"):
                raise CloudApiError("api_unavailable", "Unavailable")
            return super().request(method, path, **kwargs)

    keys = []
    for _ in range(2):
        with pytest.raises(CloudApiError) as raised:
            _deploy(UnavailableClient(project), project, tmp_path)
        keys.append(raised.value.details["retry_idempotency_key"])
    assert keys[0] == keys[1]


@pytest.mark.parametrize(
    "detail",
    [
        "Cloud's application process health check failed inside the release sandbox.",
        "Managed-egress application process health probe failed inside E2B (exit code 1).",
        "Release smoke application process health probe failed inside the MicroVM (exit code 1).",
    ],
)
def test_application_health_failure_is_not_automatically_retried(tmp_path, detail):
    project = _project(tmp_path)
    failure = _failure()
    failure["detail"] = detail
    client = Client(project, failure=failure)
    with pytest.raises(CloudApiError) as raised:
        _deploy(client, project, tmp_path)
    assert raised.value.failure["automatic_retryable"] is False
    assert not any(path.endswith("/retry") for _, path, _ in client.requests)


def test_generic_failure_contract_does_not_depend_on_server_wording(tmp_path):
    project = _project(tmp_path)
    failure = {
        key: value
        for key, value in _failure(retryable=False).items()
        if key not in {"schema_version", "diagnostic", "diagnostic_ref", "attempt"}
    }
    failure["message"] = "Repair the submitted build inputs."
    client = Client(project, failure=failure)
    with pytest.raises(CloudApiError) as raised:
        _deploy(client, project, tmp_path)
    assert raised.value.failure["code"] == "source_build_inputs_invalid"
    assert raised.value.failure["message"] == failure["message"]
