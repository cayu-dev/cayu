from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from cayu.cli import cloud
from cayu.cli._cloud_api import CloudApiError, _plain_api_error_detail
from cayu.cli._cloud_evidence import EvidenceRecorder
from cayu.cli._cloud_project import (
    CloudProcess,
    CloudProjectManifest,
    CloudSchedule,
    CloudWebProcess,
    ResolvedCloudProject,
)


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
            if method == "GET" and path.endswith("/deployments"):
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
        if method == "GET" and path.endswith("/deployments"):
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


def _legacy_source_key(project):
    # The key earlier CLI releases sent; deploying without an acknowledgement keeps it.
    identity = json.dumps(
        ["retry-agent", project.manifest.version, project.revision],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode()
    return "deploy:" + hashlib.sha256(identity).hexdigest()


def _acknowledged_deploy(client, project, tmp_path, revisions):
    return cloud._deploy(
        SimpleNamespace(
            acknowledge_breaking=revisions,
            application=None,
            no_wait=True,
            no_promote=True,
            retry_failed=True,
        ),
        client=client,
        project=project,
        recorder=EvidenceRecorder(tmp_path / "evidence"),
    )


def _requests(client, suffix):
    return [
        kwargs
        for method, path, kwargs in client.requests
        if method == "POST" and path.endswith(suffix)
    ]


def test_deploy_without_acknowledgement_keeps_payload_and_submission_key(tmp_path):
    project = _project(tmp_path)
    client = Client(project, status="accepted")
    _acknowledged_deploy(client, project, tmp_path, None)
    _acknowledged_deploy(client, project, tmp_path, [])
    creates = _requests(client, "/deployments")
    assert creates[0] == creates[1]
    assert "acknowledge_breaking" not in creates[0]["payload"]
    assert creates[0]["idempotency_key"] == _legacy_source_key(project)


def test_deploy_sends_sorted_unique_acknowledgement_under_its_own_key(tmp_path):
    project = _project(tmp_path)
    client = Client(project, status="accepted")
    _acknowledged_deploy(client, project, tmp_path, [115, 114, 115])
    _acknowledged_deploy(client, project, tmp_path, [114, 115])
    creates = _requests(client, "/deployments")
    assert creates[0]["payload"]["acknowledge_breaking"] == [114, 115]
    assert creates[0]["idempotency_key"] == creates[1]["idempotency_key"]
    assert creates[0]["idempotency_key"] != _legacy_source_key(project)
    unacknowledged = {
        key: value for key, value in creates[0]["payload"].items() if key != "acknowledge_breaking"
    }
    assert unacknowledged == project.manifest.deployment_payload(
        repository=project.repository, revision=project.revision
    )


def test_automatic_retry_of_a_replayed_failure_keeps_the_acknowledgement(tmp_path):
    project = _project(tmp_path)
    client = Client(project, status="failed")
    result = _acknowledged_deploy(client, project, tmp_path, [115])
    assert result["result"]["retry"]["deployment_id"] == "dep_new"
    assert _requests(client, "/retry")[0]["payload"] == {"acknowledge_breaking": [115]}

    plain = Client(project, status="failed")
    _acknowledged_deploy(plain, project, tmp_path, None)
    assert "payload" not in _requests(plain, "/retry")[0]


_REFUSAL = {
    "code": "storage_breaking_acknowledgement_required",
    "detail": (
        "Publishing this release migrates the Agent database from Cayu storage revision 114 "
        "to 115 across breaking revision 115, and the release does not acknowledge 115. "
        "Nothing was stopped or migrated; the previous release keeps serving."
    ),
    "hint": (
        "A breaking migration stops the previous release until this one starts, and releases "
        "built with the older Cayu can't be rolled back to afterwards. To proceed, retry "
        'release dep_old with {"acknowledge_breaking": [115]} (`cayu-cloud-operator '
        "deployment retry dep_old --application retry-agent --acknowledge-breaking 115`), or "
        "deploy again with that field. The database keeps a point-in-time backup."
    ),
    "message": (
        "The release crosses a breaking Cayu storage revision that it does not acknowledge."
    ),
}


class RefusedReleaseClient(Client):
    """The same source was deployed without the acknowledgement and Cloud refused it."""

    def __init__(self, project, *, unacknowledged_replay=True):
        super().__init__(project, status="promoted")
        self.unacknowledged_replay = unacknowledged_replay
        payload = project.manifest.deployment_payload(
            repository=project.repository, revision=project.revision
        )
        # Cloud omits the absent runtime from its canonical manifest digest.
        manifest = {key: value for key, value in payload["manifest"].items() if key != "runtime"}
        self.release = {
            "id": "dep_old",
            "version": payload["version"],
            "manifest_digest": hashlib.sha256(
                json.dumps(manifest, separators=(",", ":"), sort_keys=True).encode()
            ).hexdigest(),
            "policy_version": payload["policy_version"],
            "created_at": "2026-10-07T00:00:00Z",
            "status": "promoted",
            "acknowledge_breaking": [],
        }

    def request(self, method, path, **kwargs):
        if method == "POST" and path.endswith("/deployments"):
            self.requests.append((method, path, kwargs))
            if "acknowledge_breaking" in kwargs["payload"] or not self.unacknowledged_replay:
                raise CloudApiError(
                    "api_request_rejected",
                    "Cayu Cloud API returned HTTP 409.",
                    status_code=409,
                    detail="Application already has a deployment with this version.",
                )
            return self.release.copy()
        if method == "GET" and path.endswith("/deployments"):
            self.requests.append((method, path, kwargs))
            return {
                "items": [self.release.copy()] if self.unacknowledged_replay else [],
                "next_cursor": None,
            }
        if method == "GET" and path.endswith("/deployments/dep_old"):
            self.requests.append((method, path, kwargs))
            return {"id": "dep_old", "status": "promoted", "publication_error": _REFUSAL}
        if method == "GET" and path.endswith("/service"):
            self.requests.append((method, path, kwargs))
            return {"deployment_id": "dep_old", "status": "running", "issues": []}
        if method == "POST" and path.endswith("/dep_old/retry"):
            self.requests.append((method, path, kwargs))
            return {"id": "dep_old", "status": "promoted", "acknowledge_breaking": [115]}
        return super().request(method, path, **kwargs)


def test_acknowledged_rerun_adds_the_acknowledgement_to_the_refused_release(tmp_path):
    project = _project(tmp_path)
    client = RefusedReleaseClient(project)
    result = _acknowledged_deploy(client, project, tmp_path, [115])
    assert result["result"]["deployment"]["id"] == "dep_old"
    assert result["result"]["retry"] == {
        "acknowledge_breaking": [115],
        "deployment_id": "dep_old",
        "failure_code": "storage_breaking_acknowledgement_required",
        "previous_deployment_id": "dep_old",
        "reason": _REFUSAL["message"],
    }
    creates = _requests(client, "/deployments")
    assert len(creates) == 1
    retry = _requests(client, "/retry")
    assert len(retry) == 1
    assert retry[0]["payload"] == {"acknowledge_breaking": [115]}


def test_acknowledged_rerun_reports_the_conflict_when_no_earlier_submission_matches(tmp_path):
    project = _project(tmp_path)
    client = RefusedReleaseClient(project, unacknowledged_replay=False)
    with pytest.raises(CloudApiError) as raised:
        _acknowledged_deploy(client, project, tmp_path, [115])
    assert raised.value.status_code == 409
    assert raised.value.detail == "Application already has a deployment with this version."
    assert not _requests(client, "/retry")


def test_acknowledged_rerun_updates_the_refused_retry_instead_of_its_failed_original(tmp_path):
    project = _project(tmp_path)

    class RetriedPublicationClient(RefusedReleaseClient):
        def __init__(self):
            super().__init__(project)
            self.release["status"] = "failed"
            self.child = {
                **self.release,
                "id": "dep_child",
                "version": self.release["version"] + "-retry-1",
                "created_at": "2026-10-07T01:00:00Z",
                "status": "promoted",
            }

        def request(self, method, path, **kwargs):
            if method == "GET" and path.endswith("/deployments"):
                self.requests.append((method, path, kwargs))
                return {"items": [self.child, self.release], "next_cursor": None}
            if method == "GET" and path.endswith("/deployments/dep_child"):
                self.requests.append((method, path, kwargs))
                return {**self.child, "publication_error": _REFUSAL}
            if method == "POST" and path.endswith("/deployments/dep_child/retry"):
                self.requests.append((method, path, kwargs))
                return {**self.child, "acknowledge_breaking": [115]}
            return super().request(method, path, **kwargs)

    client = RetriedPublicationClient()
    result = _acknowledged_deploy(client, project, tmp_path, [115])
    assert result["result"]["deployment"]["acknowledge_breaking"] == [115]
    assert result["result"]["retry"]["previous_deployment_id"] == "dep_child"
    retries = [
        (path, kwargs)
        for method, path, kwargs in client.requests
        if method == "POST" and path.endswith("/retry")
    ]
    assert len(retries) == 1
    assert retries[0][0].endswith("/dep_child/retry")
    assert retries[0][1]["payload"] == {"acknowledge_breaking": [115]}


@pytest.mark.parametrize("previous_acknowledgements", [[114], [114, 116]])
def test_acknowledged_rerun_corrects_an_earlier_acknowledgement(
    tmp_path, previous_acknowledgements
):
    project = _project(tmp_path)

    class PreviouslyAcknowledgedClient(RefusedReleaseClient):
        def __init__(self):
            super().__init__(project)
            self.release["acknowledge_breaking"] = previous_acknowledgements

        def request(self, method, path, **kwargs):
            if method == "POST" and path.endswith("/deployments"):
                self.requests.append((method, path, kwargs))
                # Neither the corrected key nor the unacknowledged key created this version.
                raise CloudApiError(
                    "api_request_rejected",
                    "Cayu Cloud API returned HTTP 409.",
                    status_code=409,
                    detail="Application already has a deployment with this version.",
                )
            if method == "POST" and path.endswith("/dep_old/retry"):
                self.requests.append((method, path, kwargs))
                return {
                    **self.release,
                    "acknowledge_breaking": sorted({*previous_acknowledgements, 115}),
                }
            return super().request(method, path, **kwargs)

    client = PreviouslyAcknowledgedClient()
    result = _acknowledged_deploy(client, project, tmp_path, [115])
    assert result["result"]["deployment"]["acknowledge_breaking"] == sorted(
        {*previous_acknowledgements, 115}
    )
    assert _requests(client, "/retry")[0]["payload"] == {"acknowledge_breaking": [115]}


@pytest.mark.parametrize(
    "changes,digest",
    [
        (
            {"web": CloudWebProcess(command="python -m agent", port=8000)},
            "14d283d24742c4b3ec8e26ecc48aa057bfdda1dd474ad49646f079ba59ff22c3",
        ),
        (
            {
                "web": CloudWebProcess(
                    command="python -m agent",
                    port=8000,
                    idle_timeout_seconds=600,
                    ready_path="/health",
                    ready_timeout_seconds=3,
                    ready_start_period_seconds=60,
                    cpu_millis=1000,
                    memory_mb=2048,
                ),
                "runtime_environment": {"MODE": "worker"},
                "worker": CloudProcess(command="python -m agent.worker"),
                "schedules": (
                    CloudSchedule(
                        name="daily", command="python -m agent.daily", expression="rate(1 day)"
                    ),
                ),
            },
            "13ca696824528f9331fd0fd6a0a759001b5f5bc1ea4fc82093f4fae73540ee21",
        ),
        (
            {"worker": CloudProcess(command="python -m agent.worker")},
            "cd7ebd4790f33482001dcfd8f754e4d76ef026cc4adaf33b726ba015604eaf02",
        ),
        (
            {
                "schedules": (
                    CloudSchedule(
                        name="daily",
                        command="python -m agent.daily",
                        expression="rate(1 day)",
                        cpu_millis=1000,
                        memory_mb=2048,
                    ),
                )
            },
            "18208187f9585a89ca56fde6024020a121eca26f3d2b497e442c44a9a2f99757",
        ),
    ],
    ids=["web-defaults", "web-custom", "worker", "schedule"],
)
def test_acknowledgement_recovery_matches_cloud_runtime_manifest_digests(tmp_path, changes, digest):
    project = _project(tmp_path)
    project = replace(project, manifest=replace(project.manifest, **changes))
    client = RefusedReleaseClient(project)
    # Recorded from Cloud's DeploymentManifestPayload.to_domain() and manifest_digest(),
    # including the server's treatment of default and per-process runtime settings.
    client.release["manifest_digest"] = digest
    result = _acknowledged_deploy(client, project, tmp_path, [115])
    assert result["result"]["deployment"]["acknowledge_breaking"] == [115]


@pytest.mark.parametrize(
    "field,value",
    [("version", "different-version"), ("manifest_digest", "0" * 64), ("policy_version", "v2")],
)
def test_acknowledgement_recovery_refuses_different_submission_inputs(tmp_path, field, value):
    project = _project(tmp_path)
    client = RefusedReleaseClient(project)
    client.release[field] = value
    with pytest.raises(CloudApiError) as raised:
        _acknowledged_deploy(client, project, tmp_path, [115])
    assert raised.value.status_code == 409
    assert not _requests(client, "/retry")


def test_acknowledgement_recovery_finds_the_original_on_a_later_page(tmp_path):
    project = _project(tmp_path)

    class PagedClient(RefusedReleaseClient):
        def request(self, method, path, **kwargs):
            if (
                method == "GET"
                and path.endswith("/deployments")
                and "cursor" not in kwargs["query"]
            ):
                self.requests.append((method, path, kwargs))
                return {
                    "items": [{**self.release, "version": "another-version"}],
                    "next_cursor": "older-releases",
                }
            return super().request(method, path, **kwargs)

    client = PagedClient(project)
    result = _acknowledged_deploy(client, project, tmp_path, [115])
    assert result["result"]["deployment"]["acknowledge_breaking"] == [115]
    pages = [
        kwargs["query"]
        for method, path, kwargs in client.requests
        if method == "GET" and path.endswith("/deployments")
    ]
    assert pages == [{"limit": "100"}, {"limit": "100", "cursor": "older-releases"}]


@pytest.mark.parametrize("page", [{"items": None}, {"items": [], "next_cursor": "same"}])
def test_acknowledgement_recovery_refuses_invalid_or_cyclic_history(tmp_path, page):
    project = _project(tmp_path)

    class InvalidHistoryClient(RefusedReleaseClient):
        def request(self, method, path, **kwargs):
            if method == "GET" and path.endswith("/deployments"):
                self.requests.append((method, path, kwargs))
                return page
            return super().request(method, path, **kwargs)

    client = InvalidHistoryClient(project)
    with pytest.raises(CloudApiError) as raised:
        _acknowledged_deploy(client, project, tmp_path, [115])
    assert raised.value.category == "api_response_invalid"
    assert not _requests(client, "/retry")
    assert len(client.requests) <= 5


def test_public_retry_command_sends_the_acknowledgement(tmp_path: Path):
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
            "--acknowledge-breaking",
            "115",
        ]
    )
    cloud._deployment(arguments, client=client)
    assert client.requests[-1] == (
        "POST",
        "/v1/applications/retry-agent/deployments/dep_old/retry",
        {"idempotency_key": "one-submission", "payload": {"acknowledge_breaking": [115]}},
    )
