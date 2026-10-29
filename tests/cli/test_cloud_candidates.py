"""`cayu cloud deploy --no-promote` stages a candidate release, and `deployment discard`
drops one.

Cayu Cloud publishes a candidate only when someone promotes it while the Agent still
selects the release it was evaluated against, so the CLI must never promote it itself.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from tests.cli._cloud_candidate import stored_candidate

from cayu.cli import _cloud_project as cloud_project
from cayu.cli import cloud as cloud_cli
from cayu.cli._cloud_api import CloudApiClient, CloudApiError, _rejected_fields
from cayu.cli._cloud_evidence import EvidenceRecorder

_MANIFEST = """
schema_version = 2
application = "smoke-agent"
name = "Smoke Agent"
version = "0.1.0"
entrypoint = "python -m smoke"
capabilities = ["model.generate"]
cpu_millis = 512
memory_mb = 1024
timeout_seconds = 600
environment = "python"
compatibility = "cayu>=0.1"
policy_version = "v1"
[web]
command = "python -m smoke.web"
port = 8000
"""
_SNAPSHOT_ROOT = "5" * 64


def _project(tmp_path: Path) -> cloud_project.ResolvedCloudProject:
    return cloud_project.ResolvedCloudProject(
        root=None,
        manifest_path=tmp_path / "cayu-cloud.toml",
        manifest=cloud_project.CloudProjectManifest.loads(_MANIFEST),
        repository="https://github.com/example/smoke-agent",
        revision="c" * 40,
    )


class CandidateCloud:
    """Cloud that stages a candidate release, building it through ``statuses``."""

    def __init__(
        self,
        *,
        selected: str | None = "dep_live",
        statuses: tuple[str, ...] = ("image_built", "smoke_tested"),
        staged_as_candidate: bool = True,
    ) -> None:
        self.selected = selected
        self.statuses = list(statuses)
        self.staged_as_candidate = staged_as_candidate
        self.requests: list[tuple[str, str, dict[str, Any]]] = []
        self.candidate: dict[str, Any] | None = None

    def _release(self, status: str) -> dict[str, Any]:
        release: dict[str, Any] = {"id": "dep_candidate", "status": status}
        if self.staged_as_candidate:
            release["candidate"] = stored_candidate(self.candidate or {})
        return release

    def request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        self.requests.append((method, path, kwargs))
        if path == "/v1/applications":
            return {
                "items": [
                    {
                        "current_deployment_id": self.selected,
                        "id": "smoke-agent",
                        "name": "Smoke Agent",
                        "revision": 7,
                    }
                ]
            }
        if method == "POST" and path == "/v1/applications/smoke-agent/deployments":
            self.candidate = kwargs["payload"].get("candidate")
            return self._release("accepted")
        if method == "GET" and path == "/v1/applications/smoke-agent/deployments/dep_candidate":
            status = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
            return self._release(status)
        if method == "POST" and path.endswith("/dep_candidate/cancel"):
            return {"deployment_id": "dep_candidate", "requested_action": "cancel"}
        if method == "GET" and path == "/v1/applications/smoke-agent":
            return {"current_deployment_id": self.selected, "id": "smoke-agent", "revision": 7}
        raise AssertionError(f"unexpected Cloud request: {method} {path}")

    def created(self) -> dict[str, Any]:
        (create,) = [
            kwargs
            for method, path, kwargs in self.requests
            if method == "POST" and path.endswith("/deployments")
        ]
        return create

    def published(self) -> list[tuple[str, str]]:
        return [
            (method, path)
            for method, path, _ in self.requests
            # A read of the service (deploy's running-task check) publishes nothing.
            if method != "GET" and (path.endswith("/promote") or path.endswith("/service"))
        ]


_STAGE = ("--no-promote", "--evaluated-against", "dep_live")


def _deploy(client: CandidateCloud, tmp_path: Path, *flags: str) -> dict[str, Any]:
    arguments = cloud_cli._build_parser().parse_args(["deploy", ".", *flags])
    cloud_cli._preflight_deploy_wait(arguments)
    clock = iter(range(0, 10_000, 5))
    return cloud_cli._deploy(
        arguments,
        client=client,  # type: ignore[arg-type]
        recorder=EvidenceRecorder(tmp_path / "evidence"),
        project=_project(tmp_path),
        sleep=lambda _: None,
        monotonic=lambda: float(next(clock)),
    )


def test_no_promote_stages_a_candidate_against_the_stated_release(tmp_path: Path) -> None:
    client = CandidateCloud()

    result = _deploy(client, tmp_path, *_STAGE)

    create = client.created()
    assert create["payload"]["candidate"] == {"evaluated_against_deployment_id": "dep_live"}
    # The candidate is part of the submission, so it never replays a plain deploy's release.
    project = _project(tmp_path)
    plain_key = cloud_cli._deployment_idempotency_key(
        application_id="smoke-agent",
        version=project.manifest.version,
        revision=project.revision,
    )
    assert create["idempotency_key"] != plain_key
    deployment = result["result"]["deployment"]
    assert (deployment["status"], deployment["candidate"]["evaluated_against_deployment_id"]) == (
        "smoke_tested",
        "dep_live",
    )
    assert client.published() == []
    assert result["result"]["service"] is None
    assert result["result"]["service_publication_pending"] is False


def test_no_promote_records_the_evaluation_it_was_accepted_on(tmp_path: Path) -> None:
    client = CandidateCloud(selected="dep_baseline")

    _deploy(
        client,
        tmp_path,
        "--no-promote",
        "--no-wait",
        "--evaluated-against",
        "dep_baseline",
        "--expires-in-seconds",
        "3600",
        "--snapshot-root",
        _SNAPSHOT_ROOT,
        "--brief",
        "brief-monthend-cost",
        "--requirement",
        "req-2",
        "--requirement",
        "req-1",
        "--requirement",
        "req-2",
        "--receipt",
        "evaluation-2=" + "b" * 64,
        "--receipt",
        "evaluation-1=" + "a" * 64,
    )

    assert client.created()["payload"]["candidate"] == {
        "brief_id": "brief-monthend-cost",
        "evaluated_against_deployment_id": "dep_baseline",
        "expires_in_seconds": 3600,
        "receipts": [
            {"fingerprint": "a" * 64, "receipt_id": "evaluation-1"},
            {"fingerprint": "b" * 64, "receipt_id": "evaluation-2"},
        ],
        "requirement_ids": ["req-1", "req-2"],
        "snapshot_root": _SNAPSHOT_ROOT,
    }
    assert client.published() == []


def test_a_new_agent_stages_a_candidate_evaluated_against_no_release(tmp_path: Path) -> None:
    client = CandidateCloud(selected=None)

    _deploy(client, tmp_path, "--no-promote", "--no-wait", "--evaluated-against", "none")

    assert client.created()["payload"]["candidate"] == {"evaluated_against_deployment_id": None}


def test_the_baseline_is_never_guessed(tmp_path: Path) -> None:
    arguments = cloud_cli._build_parser().parse_args(["deploy", ".", "--no-promote"])

    with pytest.raises(cloud_cli.CloudCommandError) as raised:
        cloud_cli._preflight_deploy_wait(arguments)

    assert raised.value.category == "invalid_input"
    assert "--evaluated-against" in str(raised.value)


def test_the_key_of_a_candidate_names_its_baseline_not_the_selection(tmp_path: Path) -> None:
    staged = CandidateCloud(selected="dep_live")
    _deploy(staged, tmp_path, *_STAGE, "--no-wait")
    # A rerun after the selection changed sends the same request, so Cloud replays it.
    rerun = CandidateCloud(selected="dep_other")
    with pytest.raises(cloud_cli.CloudCommandError):
        _deploy(rerun, tmp_path, *_STAGE, "--no-wait")
    assert rerun.created()["idempotency_key"] == staged.created()["idempotency_key"]
    other = CandidateCloud(selected="dep_other")
    _deploy(other, tmp_path, "--no-promote", "--no-wait", "--evaluated-against", "dep_other")
    assert other.created()["idempotency_key"] != staged.created()["idempotency_key"]


def test_a_candidate_whose_baseline_is_no_longer_selected_is_refused(tmp_path: Path) -> None:
    # A rerun replays the candidate staged against dep_live; dep_other was selected since.
    client = CandidateCloud(selected="dep_other")

    with pytest.raises(cloud_cli.CloudCommandError) as raised:
        _deploy(client, tmp_path, *_STAGE)

    assert raised.value.category == "candidate_baseline_changed"
    assert "dep_candidate" in str(raised.value)
    assert "now selects dep_other" in str(raised.value)
    # The Agent was read again before refusing.
    assert ("GET", "/v1/applications/smoke-agent") in [
        (method, path) for method, path, _ in client.requests
    ]
    assert client.published() == []


def test_a_release_that_is_not_a_candidate_is_never_promoted(tmp_path: Path) -> None:
    # Cloud already holds a plain release of this source, which it publishes on its own.
    client = CandidateCloud(staged_as_candidate=False, statuses=("smoke_tested",))

    with pytest.raises(cloud_cli.CloudCommandError) as raised:
        _deploy(client, tmp_path, *_STAGE)

    assert raised.value.category == "candidate_conflict"
    assert "dep_candidate" in str(raised.value)
    assert client.published() == []


@pytest.mark.parametrize(
    "flags",
    [
        ("--evaluated-against", "dep_baseline"),
        ("--snapshot-root", _SNAPSHOT_ROOT),
        ("--brief", "brief-1"),
        ("--requirement", "req-1"),
        ("--receipt", "evaluation-1=" + "a" * 64),
        ("--expires-in-seconds", "3600"),
    ],
)
def test_candidate_flags_need_no_promote(flags: tuple[str, str]) -> None:
    arguments = cloud_cli._build_parser().parse_args(["deploy", ".", *flags])

    with pytest.raises(cloud_cli.CloudCommandError) as raised:
        cloud_cli._preflight_deploy_wait(arguments)

    assert raised.value.category == "invalid_input"
    assert "--no-promote" in str(raised.value)


@pytest.mark.parametrize(
    "flags",
    [
        ("--evaluated-against", "release-1"),
        ("--evaluated-against", "None"),
        ("--snapshot-root", "sha256:" + "5" * 57),
        ("--snapshot-root", "A" * 64),
        ("--brief", "../brief"),
        ("--requirement", "req 1"),
        ("--receipt", "evaluation-1"),
        ("--receipt", "evaluation-1=" + "z" * 64),
        ("--expires-in-seconds", "3599"),
        ("--expires-in-seconds", "2592001"),
    ],
)
def test_invalid_candidate_flags_are_refused(flags: tuple[str, str]) -> None:
    with pytest.raises(cloud_cli.CloudCommandError) as raised:
        cloud_cli._build_parser().parse_args(["deploy", ".", "--no-promote", *flags])

    assert raised.value.category == "invalid_input"


def test_a_receipt_given_with_two_fingerprints_is_refused() -> None:
    arguments = cloud_cli._build_parser().parse_args(
        [
            "deploy",
            ".",
            *_STAGE,
            "--receipt",
            "evaluation-1=" + "a" * 64,
            "--receipt",
            "evaluation-1=" + "b" * 64,
        ]
    )

    with pytest.raises(cloud_cli.CloudCommandError) as raised:
        cloud_cli._preflight_deploy_wait(arguments)

    assert raised.value.category == "invalid_input"


def _discard(client: CandidateCloud) -> dict[str, Any]:
    return cloud_cli._deployment(
        SimpleNamespace(
            application="smoke-agent",
            deployment_command="discard",
            deployment_id="dep_candidate",
        ),
        client=client,  # type: ignore[arg-type]
    )


def test_discard_cancels_a_candidate(tmp_path: Path) -> None:
    client = CandidateCloud(statuses=("smoke_tested",))
    client.candidate = {"evaluated_against_deployment_id": "dep_live"}

    result = _discard(client)

    assert result == {
        "operation": "deployment.discard",
        "result": {"deployment_id": "dep_candidate", "requested_action": "cancel"},
    }
    assert client.requests[-1][0:2] == (
        "POST",
        "/v1/applications/smoke-agent/deployments/dep_candidate/cancel",
    )


def test_discard_refuses_a_release_that_is_not_a_candidate() -> None:
    client = CandidateCloud(staged_as_candidate=False, statuses=("smoke_tested",))

    with pytest.raises(cloud_cli.CloudCommandError) as raised:
        _discard(client)

    assert raised.value.category == "not_a_candidate"
    assert not any(path.endswith("/cancel") for _, path, _ in client.requests)


def _listed_release(project: cloud_project.ResolvedCloudProject, **fields: Any) -> dict[str, Any]:
    """This source's release as Cloud lists it: same version, manifest and policy."""

    payload = project.manifest.deployment_payload(
        repository=project.repository, revision=project.revision
    )
    manifest = dict(payload["manifest"])
    runtime = manifest.pop("runtime", None)
    if runtime is not None:
        # Cloud's canonical manifest includes a web process's unset idle timeout.
        manifest["runtime"] = {**runtime, "web": {"idle_timeout_seconds": None, **runtime["web"]}}
    return {
        "id": "dep_existing",
        "version": payload["version"],
        "manifest_digest": hashlib.sha256(
            json.dumps(manifest, separators=(",", ":"), sort_keys=True).encode()
        ).hexdigest(),
        "policy_version": payload["policy_version"],
        "status": "smoke_tested",
        "acknowledge_breaking": [],
        "session_policy": {"acknowledge_sessions": [], "mode": "wait", "wait_seconds": 900},
        **fields,
    }


_SMOKE_FAILURE = {
    "automatic_retryable": True,
    "code": "release_smoke_failed",
    "detail": "The release smoke test did not complete successfully.",
    "hint": "Inspect the deployment diagnostics.",
    "message": "The release smoke test failed.",
    "phase": "smoke_tested",
}


def _timeline(last_step: str) -> dict[str, Any]:
    """A timeline whose `agent_ready` step ended ``last_step``, with an earlier smoke failure
    Cloud retried on its own."""

    return {
        "failure": _SMOKE_FAILURE,
        "items": [
            {"key": "release_selected", "state": last_step},
            {"key": "agent_ready", "state": last_step},
        ],
    }


def _baseline_refusal(message: str) -> CloudApiError:
    return CloudApiError(
        "api_request_rejected",
        f"Cayu Cloud API returned HTTP 409: {message}",
        status_code=409,
        detail=message,
        code="candidate_baseline_changed",
    )


class ConflictingCloud(CandidateCloud):
    """Cloud whose create conflicts, with ``code`` when given, and lists ``existing``.

    ``timeline`` is the existing release's, and ``retry`` what a retry of it answers (a
    new attempt) or raises.
    """

    def __init__(
        self,
        existing: dict[str, Any] | None,
        *,
        selected: str | None = "dep_live",
        code: str | None = None,
        timeline: dict[str, Any] | None = None,
        retry: CloudApiError | None = None,
    ) -> None:
        super().__init__(selected=selected)
        self.existing = existing
        self.code = code
        self.timeline = timeline
        self.retry = retry

    def request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        if method == "POST" and path == "/v1/applications/smoke-agent/deployments":
            self.requests.append((method, path, kwargs))
            if self.code == "candidate_baseline_changed":
                raise _baseline_refusal("The Agent selects dep_other, but the candidate was ...")
            raise CloudApiError(
                "api_request_rejected",
                "Cayu Cloud API returned HTTP 409.",
                status_code=409,
                detail="Application already has a deployment with this version.",
                code=self.code,
            )
        if path.endswith("/dep_existing/timeline"):
            self.requests.append((method, path, kwargs))
            return self.timeline or {"failure": None, "items": []}
        if method == "POST" and path.endswith("/dep_existing/retry"):
            self.requests.append((method, path, kwargs))
            if self.retry is not None:
                raise self.retry
            self.candidate = {"evaluated_against_deployment_id": "dep_live"}
            return self._release("accepted")
        if method == "GET" and path == "/v1/applications/smoke-agent/deployments":
            self.requests.append((method, path, kwargs))
            return {"items": [] if self.existing is None else [dict(self.existing)]}
        if method == "GET" and path == "/v1/applications/smoke-agent/deployments/dep_existing":
            self.requests.append((method, path, kwargs))
            assert self.existing is not None
            return dict(self.existing)
        return super().request(method, path, **kwargs)

    def listed(self) -> bool:
        return any(
            method == "GET" and path.endswith("/deployments") for method, path, _ in self.requests
        )


@pytest.mark.parametrize(
    "flags,stored",
    [
        # Another evaluation of the same source.
        (("--brief", "brief-2"), {"brief_id": "brief-1"}),
        # Another baseline.
        ((), {"evaluated_against_deployment_id": "dep_earlier"}),
        # Another promotion window.
        (("--expires-in-seconds", "3600"), {}),
    ],
)
def test_a_candidate_with_another_statement_is_not_taken_up(
    tmp_path: Path, flags: tuple[str, ...], stored: dict[str, Any]
) -> None:
    existing = _listed_release(
        _project(tmp_path),
        candidate=stored_candidate({"evaluated_against_deployment_id": "dep_live", **stored}),
    )
    client = ConflictingCloud(existing)

    with pytest.raises(cloud_cli.CloudCommandError) as raised:
        _deploy(client, tmp_path, *_STAGE, "--session-policy", "block", *flags)

    assert raised.value.category == "candidate_conflict"
    assert "dep_existing" in str(raised.value)
    assert client.published() == []


def test_a_conflict_no_release_explains_is_reported_without_reading_history(
    tmp_path: Path,
) -> None:
    client = ConflictingCloud(None, code="application_archived")

    with pytest.raises(CloudApiError) as raised:
        _deploy(client, tmp_path, *_STAGE)

    assert raised.value.code == "application_archived"
    assert not client.listed()


def test_cloud_refusing_a_candidate_is_reported_as_the_cli_reports_it(tmp_path: Path) -> None:
    # The same refusal, whether Cloud or the CLI finds it, has one category.
    client = ConflictingCloud(None, code="candidate_baseline_changed")

    with pytest.raises(cloud_cli.CloudCommandError) as raised:
        _deploy(client, tmp_path, *_STAGE)

    assert raised.value.category == "candidate_baseline_changed"
    assert str(raised.value) == "The Agent selects dep_other, but the candidate was ..."
    assert not client.listed()


def test_a_plain_deploy_never_takes_up_a_candidate(tmp_path: Path) -> None:
    existing = _listed_release(
        _project(tmp_path),
        candidate=stored_candidate({"evaluated_against_deployment_id": "dep_live"}),
    )
    client = ConflictingCloud(existing)

    with pytest.raises(cloud_cli.CloudCommandError) as raised:
        _deploy(client, tmp_path, "--session-policy", "block")

    assert raised.value.category == "candidate_conflict"
    assert "cayu cloud deployment promote dep_existing --application smoke-agent" in str(
        raised.value
    )
    assert client.published() == []
    assert not any(path.endswith("/retry") for _, path, _ in client.requests)


def test_a_rerun_of_a_candidate_never_publishes_it(tmp_path: Path) -> None:
    # Someone promoted it since, so the Agent selects it; staging again only reports it.
    existing = _listed_release(
        _project(tmp_path),
        status="promoted",
        candidate=stored_candidate({"evaluated_against_deployment_id": "dep_live"}),
    )
    client = ConflictingCloud(existing, selected="dep_existing")

    result = _deploy(client, tmp_path, *_STAGE, "--acknowledge-breaking", "115")

    assert result["result"]["deployment"]["status"] == "promoted"
    assert result["result"]["service"] is None
    assert result["result"]["service_publication_pending"] is False
    # The acknowledgement would need a publication retry, which only a person runs.
    assert result["result"]["not_applied"]["acknowledge_breaking"] == [115]
    # Reading the Agent's running tasks for the candidate's later promotion publishes nothing.
    assert not any(
        path.endswith(("/retry", "/promote", "/service"))
        for method, path, _ in client.requests
        if method != "GET"
    )


@pytest.mark.parametrize("status", ["cancelled", "destroyed"])
def test_a_rerun_of_a_discarded_candidate_says_so(tmp_path: Path, status: str) -> None:
    # Destroyed an hour after it was discarded, it still shows an earlier smoke failure that
    # Cloud retried on its own; its timeline ends cancelled.
    existing = _listed_release(
        _project(tmp_path),
        status=status,
        candidate=stored_candidate({"evaluated_against_deployment_id": "dep_live"}),
    )
    client = ConflictingCloud(existing, timeline=_timeline("cancelled"))

    with pytest.raises(cloud_cli.CloudCommandError) as raised:
        _deploy(client, tmp_path, *_STAGE, "--session-policy", "block")

    assert raised.value.category == "candidate_discarded"
    assert "dep_existing" in str(raised.value)
    assert not any(path.endswith(("/retry", "/promote")) for _, path, _ in client.requests)


def test_a_candidate_destroyed_after_a_failed_build_is_retried(tmp_path: Path) -> None:
    existing = _listed_release(
        _project(tmp_path),
        status="destroyed",
        candidate=stored_candidate({"evaluated_against_deployment_id": "dep_live"}),
    )
    client = ConflictingCloud(existing, timeline=_timeline("failed"))

    result = _deploy(client, tmp_path, *_STAGE, "--session-policy", "block", "--no-wait")

    assert result["result"]["retry"]["previous_deployment_id"] == "dep_existing"
    assert client.published() == []


def test_a_failed_candidate_build_is_not_retried_after_its_baseline_changed(
    tmp_path: Path,
) -> None:
    existing = _listed_release(
        _project(tmp_path),
        status="failed",
        candidate=stored_candidate({"evaluated_against_deployment_id": "dep_live"}),
    )
    client = ConflictingCloud(existing, selected="dep_other", timeline=_timeline("failed"))

    with pytest.raises(cloud_cli.CloudCommandError) as raised:
        _deploy(client, tmp_path, *_STAGE, "--session-policy", "block")

    assert raised.value.category == "candidate_baseline_changed"
    assert not any(path.endswith("/retry") for _, path, _ in client.requests)


def test_a_retry_cloud_refuses_for_a_changed_baseline_keeps_that_category(
    tmp_path: Path,
) -> None:
    # The selection changes between the deploy's check and its retry.
    existing = _listed_release(
        _project(tmp_path),
        status="failed",
        candidate=stored_candidate({"evaluated_against_deployment_id": "dep_live"}),
    )
    client = ConflictingCloud(
        existing,
        timeline=_timeline("failed"),
        retry=_baseline_refusal("The Agent selects dep_other, but the candidate was ..."),
    )

    with pytest.raises(cloud_cli.CloudCommandError) as raised:
        _deploy(client, tmp_path, *_STAGE, "--session-policy", "block")

    assert raised.value.category == "candidate_baseline_changed"
    assert str(raised.value) == "The Agent selects dep_other, but the candidate was ..."


@pytest.mark.parametrize("status", ["cancelled", "destroyed"])
def test_a_candidate_discarded_while_the_deploy_waits_says_so(tmp_path: Path, status: str) -> None:
    class DiscardingCloud(CandidateCloud):
        def request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
            if path.endswith("/dep_candidate/timeline"):
                return _timeline("cancelled")
            return super().request(method, path, **kwargs)

    client = DiscardingCloud(statuses=("image_built", status))

    with pytest.raises(cloud_cli.CloudCommandError) as raised:
        _deploy(client, tmp_path, *_STAGE)

    assert raised.value.category == "candidate_discarded"
    assert "dep_candidate" in str(raised.value)


def test_a_candidate_refusal_keeps_clouds_message_as_its_detail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    message = "The candidate expired at 2026-10-23T12:00:00+00:00. Stage a new candidate."
    monkeypatch.setattr(
        CloudApiClient,
        "_send",
        lambda *_, **__: httpx.Response(
            409, json={"detail": {"code": "candidate_expired", "message": message}}
        ),
    )

    with pytest.raises(CloudApiError) as raised:
        CloudApiClient(api_url="https://cloud.example.test", api_key="key").request(
            "POST", "/v1/applications/smoke-agent/deployments/dep_candidate/promote"
        )

    assert (raised.value.code, raised.value.detail) == ("candidate_expired", message)


class RejectingCloud(CandidateCloud):
    """Cloud that rejects a create with HTTP 422 naming ``fields``."""

    def __init__(self, fields: tuple[tuple[str, str], ...]) -> None:
        super().__init__()
        self.fields = fields

    def request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        if method == "POST" and path == "/v1/applications/smoke-agent/deployments":
            raise CloudApiError(
                "api_request_rejected",
                "Cayu Cloud API returned HTTP 422.",
                status_code=422,
                rejected_fields=self.fields,
            )
        return super().request(method, path, **kwargs)


def test_a_cloud_without_candidate_releases_is_named_as_the_reason(tmp_path: Path) -> None:
    with pytest.raises(cloud_cli.CloudCommandError) as unsupported:
        _deploy(RejectingCloud((("body.candidate", "extra_forbidden"),)), tmp_path, *_STAGE)
    assert unsupported.value.category == "candidate_unsupported"
    assert "nothing was deployed" in str(unsupported.value)

    # A Cloud that knows candidates but rejects this statement says so instead.
    with pytest.raises(cloud_cli.CloudCommandError) as invalid:
        _deploy(RejectingCloud((("body.candidate", "value_error"),)), tmp_path, *_STAGE)
    assert invalid.value.category == "candidate_invalid"

    with pytest.raises(CloudApiError) as other:
        _deploy(
            RejectingCloud((("body.manifest.entrypoint", "string_too_short"),)),
            tmp_path,
            *_STAGE,
        )
    assert other.value.status_code == 422


def test_a_validation_rejection_names_only_plain_field_locations() -> None:
    response = httpx.Response(
        422,
        json={
            "detail": [
                {"loc": ["body", "candidate"], "msg": "Extra inputs", "type": "extra_forbidden"},
                {"loc": ["body", "manifest", "resources", 0], "msg": "x", "type": "int_type"},
                {"loc": ["body", "secret value"], "msg": "x", "type": "value_error"},
                {"loc": ["body", "brief"], "msg": "x", "type": "Not A Type"},
                {"msg": "no location", "type": "value_error"},
            ]
        },
    )

    assert _rejected_fields(response) == (
        ("body.candidate", "extra_forbidden"),
        ("body.manifest.resources.0", "int_type"),
    )
    assert _rejected_fields(httpx.Response(409, json={"detail": []})) == ()


def test_discard_reports_cloud_refusing_a_candidate_being_published() -> None:
    class PublishingCloud(CandidateCloud):
        def request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
            if path.endswith("/cancel"):
                raise CloudApiError(
                    "api_request_rejected",
                    "Cayu Cloud API returned HTTP 409: Release dep_candidate is being "
                    "published after its promotion, so it can't be cancelled.",
                    status_code=409,
                )
            return super().request(method, path, **kwargs)

    client = PublishingCloud(statuses=("smoke_tested",))
    client.candidate = {"evaluated_against_deployment_id": "dep_live"}

    with pytest.raises(CloudApiError) as raised:
        _discard(client)

    assert raised.value.status_code == 409
    assert "being published" in str(raised.value)
