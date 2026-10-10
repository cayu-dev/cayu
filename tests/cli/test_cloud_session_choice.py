"""`cayu cloud` session-aware deploys: the owner's choice for unfinished sessions."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from cayu.cli import cloud, main
from cayu.cli._cloud_api import CloudApiError
from cayu.cli._cloud_evidence import EvidenceRecorder
from cayu.cli._cloud_project import CloudProjectManifest, ResolvedCloudProject

_DEPLOYMENT = "dep_0123456789abcdef0123456789abcdef"
_RETRY = "dep_retry"


def _project(tmp_path, *, web=False):
    """A source-bundle project; `web` gives it a web process, so Cloud publishes it."""

    web_runtime = '\n[web]\ncommand = "python -m agent.web"\nport = 8000\n' if web else ""
    manifest = CloudProjectManifest.loads(
        f"""
schema_version = {2 if web else 1}
application = "session-agent"
name = "Session Agent"
version = "1.0.0"
entrypoint = "python -m agent"
capabilities = ["model.generate"]
cpu_millis = 512
memory_mb = 1024
timeout_seconds = 600
environment = "python"
compatibility = "cayu>=0.1"
policy_version = "v1"
{web_runtime}"""
    )
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


def _cloud_choice(payload):
    """The session choice Cloud stores for a create or retry body."""

    sessions = (payload or {}).get("acknowledge_sessions", [])
    return {
        "acknowledge_sessions": sessions,
        "mode": (payload or {}).get("session_policy") or ("proceed" if sessions else "wait"),
        "wait_seconds": (payload or {}).get("session_wait_seconds", 900),
    }


def _legacy_source_key(project):
    identity = json.dumps(
        ["session-agent", project.manifest.version, project.revision],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode()
    return "deploy:" + hashlib.sha256(identity).hexdigest()


class Client:
    """Accepts a deploy; replays the create with `status`."""

    def __init__(self, project, *, status="accepted"):
        self.project = project
        self.status = status
        self.requests = []

    def upload_bytes(self, url, bundle):
        pass

    def request(self, method, path, **kwargs):
        self.requests.append((method, path, kwargs))
        if path == "/v1/applications":
            return {"items": [{"id": "session-agent", "name": "Session Agent", "revision": 1}]}
        if path == "/v1/source-bundles/uploads":
            return {
                "content_digest": self.project.content_digest,
                "repository": self.project.repository,
                "revision": self.project.revision,
                "size_bytes": len(self.project.bundle),
                "upload_url": "https://uploads.example.test/signed",
            }
        if method == "POST" and path.endswith("/deployments"):
            return {
                "acknowledge_breaking": kwargs["payload"].get("acknowledge_breaking", []),
                "id": "dep_old",
                "status": self.status,
                "session_policy": _cloud_choice(kwargs["payload"]),
            }
        if path.endswith("/timeline"):
            return {
                "failure": {
                    "automatic_retryable": True,
                    "code": "release_smoke_failed",
                    "detail": "The release smoke test did not complete successfully.",
                    "hint": "Inspect the deployment diagnostics.",
                    "message": "The release smoke test failed.",
                    "phase": "smoke_tested",
                }
            }
        if method == "POST" and path.endswith("/retry"):
            return {
                "id": "dep_new",
                "status": "accepted",
                "session_policy": _cloud_choice(kwargs.get("payload")),
            }
        raise AssertionError((method, path))


def _deploy(client, project, tmp_path, *flags, wait=False):
    parsed = cloud._build_parser().parse_args(["deploy", ".", *flags])
    clock = _Clock()
    return cloud._deploy(
        SimpleNamespace(
            acknowledge_breaking=parsed.acknowledge_breaking,
            acknowledge_session=parsed.acknowledge_session,
            application=None,
            no_wait=not wait,
            no_promote=not wait,
            poll_seconds=5.0,
            retry_failed=True,
            session_policy=parsed.session_policy,
            session_wait_seconds=parsed.session_wait_seconds,
            wait_seconds=60.0,
        ),
        client=client,
        project=project,
        recorder=EvidenceRecorder(tmp_path / "evidence"),
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )


def _posts(client, suffix):
    return [
        kwargs
        for method, path, kwargs in client.requests
        if method == "POST" and path.endswith(suffix)
    ]


_ACKNOWLEDGED_S1 = {
    "policy": "proceed",
    "read": "complete",
    "sessions": [
        {"acknowledged": True, "blocking": True, "id": "s1"},
        {"acknowledged": False, "blocking": False, "id": "s-idle"},
    ],
    "state": "acknowledged",
    "truncated": False,
}


_BLOCKED = {"read": "complete", "state": "blocked", "truncated": False}


# Command-line validation


@pytest.mark.parametrize(
    "command",
    [
        ["deploy", "."],
        ["deployment", "retry", "dep_one", "--application", "session-agent"],
        ["rollback", "dep_one", "--application", "session-agent"],
    ],
)
@pytest.mark.parametrize(
    "flags,mentions",
    [
        (["--session-policy", "later"], "--session-policy"),
        (["--session-wait-seconds", "59"], "--session-wait-seconds"),
        (["--session-wait-seconds", "3601"], "--session-wait-seconds"),
        (["--session-wait-seconds", "900.5"], "--session-wait-seconds"),
        (["--session-wait-seconds", "abc"], "--session-wait-seconds"),
        (["--acknowledge-session="], "--acknowledge-session"),
        (["--acknowledge-session", " s1"], "--acknowledge-session"),
        (["--acknowledge-session", "s\x01"], "--acknowledge-session"),
        (["--acknowledge-session", "s" * 257], "--acknowledge-session"),
        (["--session-policy", "proceed"], "--acknowledge-session"),
        (["--session-policy", "block", "--acknowledge-session", "s1"], "--session-policy"),
        (["--session-policy", "wait", "--acknowledge-session", "*"], "--session-policy"),
        (["--session-policy", "block", "--session-wait-seconds", "120"], "applies only"),
        (["--acknowledge-session", "s1", "--session-wait-seconds", "120"], "applies only"),
    ],
)
def test_session_choice_is_validated_before_authentication(
    command, flags, mentions, monkeypatch, capsys
):
    def unexpected_client(*_args, **_kwargs):
        raise AssertionError("an invalid session choice reached Cloud authentication")

    monkeypatch.setattr(cloud, "_cloud_client", unexpected_client)
    assert main(["cloud", *command, *flags]) == 2
    error = json.loads(capsys.readouterr().out)["error"]
    assert error["category"] == "invalid_input"
    assert mentions in error["message"]


def test_session_choice_limits_the_acknowledged_sessions(monkeypatch, capsys):
    monkeypatch.setattr(cloud, "_cloud_client", lambda *_a, **_k: pytest.fail("authenticated"))
    flags = [item for index in range(201) for item in ("--acknowledge-session", f"s{index}")]
    assert main(["cloud", "deployment", "retry", "dep_one", "--application", "a", *flags]) == 2
    error = json.loads(capsys.readouterr().out)["error"]
    assert error["category"] == "invalid_input"
    assert "At most 200 distinct --acknowledge-session" in error["message"]


@pytest.mark.parametrize(
    "flags,choice",
    [
        ([], None),
        (
            ["--session-policy", "wait"],
            {"acknowledge_sessions": [], "mode": "wait", "wait_seconds": 900},
        ),
        (
            ["--session-wait-seconds", "60"],
            {"acknowledge_sessions": [], "mode": "wait", "wait_seconds": 60},
        ),
        (
            ["--session-policy", "block"],
            {"acknowledge_sessions": [], "mode": "block", "wait_seconds": 900},
        ),
        (
            ["--acknowledge-session", "s2", "--acknowledge-session", "s1"] * 2,
            {"acknowledge_sessions": ["s1", "s2"], "mode": "proceed", "wait_seconds": 900},
        ),
        (
            [
                "--session-policy",
                "proceed",
                "--acknowledge-session",
                "s1",
                "--acknowledge-session",
                "*",
            ],
            {"acknowledge_sessions": ["*"], "mode": "proceed", "wait_seconds": 900},
        ),
    ],
)
def test_session_choice_normalizes_like_cloud(flags, choice):
    parsed = cloud._build_parser().parse_args(["deploy", ".", *flags])
    assert cloud._session_choice(parsed) == choice


# Request bodies and idempotency


@pytest.mark.parametrize("flags", [[], ["--session-policy", "wait"]])
def test_deploy_with_the_default_choice_keeps_payload_and_submission_key(tmp_path, flags):
    project = _project(tmp_path)
    client = Client(project)
    _deploy(client, project, tmp_path, *flags)
    create = _posts(client, "/deployments")[0]
    assert create["idempotency_key"] == _legacy_source_key(project)
    assert create["payload"] == project.manifest.deployment_payload(
        repository=project.repository, revision=project.revision
    )


@pytest.mark.parametrize(
    "flags,fields",
    [
        (["--session-policy", "block"], {"session_policy": "block"}),
        (
            ["--session-wait-seconds", "1800"],
            {"session_policy": "wait", "session_wait_seconds": 1800},
        ),
        (
            ["--acknowledge-session", "s2", "--acknowledge-session", "s1"],
            {"session_policy": "proceed", "acknowledge_sessions": ["s1", "s2"]},
        ),
        (
            ["--acknowledge-session", "s1", "--acknowledge-session", "*"],
            {"session_policy": "proceed", "acknowledge_sessions": ["*"]},
        ),
    ],
)
def test_deploy_sends_a_non_default_choice_under_its_own_key(tmp_path, flags, fields):
    project = _project(tmp_path)
    client = Client(project)
    _deploy(client, project, tmp_path, *flags)
    _deploy(client, project, tmp_path, *flags)
    first, second = _posts(client, "/deployments")
    assert first == second
    assert first["idempotency_key"] != _legacy_source_key(project)
    base = project.manifest.deployment_payload(
        repository=project.repository, revision=project.revision
    )
    assert first["payload"] == {**base, **fields}


def test_deploy_key_covers_the_choice_and_the_acknowledgement_independently(tmp_path):
    project = _project(tmp_path)
    keys = set()
    for flags in (
        [],
        ["--session-policy", "block"],
        ["--acknowledge-session", "s1"],
        ["--acknowledge-session", "s2"],
        ["--acknowledge-breaking", "115"],
        ["--acknowledge-breaking", "115", "--session-policy", "block"],
    ):
        client = Client(project)
        _deploy(client, project, tmp_path, *flags)
        keys.add(_posts(client, "/deployments")[0]["idempotency_key"])
    assert len(keys) == 6

    # Order and duplicates of the acknowledged sessions don't change the key.
    one, other = Client(project), Client(project)
    _deploy(one, project, tmp_path, "--acknowledge-session", "a", "--acknowledge-session", "b")
    _deploy(
        other, project, tmp_path, *["--acknowledge-session", "b", "--acknowledge-session", "a"] * 2
    )
    assert _posts(one, "/deployments") == _posts(other, "/deployments")


def test_automatic_retry_of_a_replayed_failure_keeps_the_choice(tmp_path):
    project = _project(tmp_path)
    client = Client(project, status="failed")
    _deploy(client, project, tmp_path, "--acknowledge-breaking", "115", "--session-policy", "block")
    assert _posts(client, "/retry")[0]["payload"] == {
        "acknowledge_breaking": [115],
        "session_policy": "block",
    }


@pytest.mark.parametrize(
    "flags,payload",
    [
        ([], None),
        (["--session-policy", "wait"], {"session_policy": "wait"}),
        (
            ["--session-policy", "wait", "--session-wait-seconds", "3600"],
            {"session_policy": "wait", "session_wait_seconds": 3600},
        ),
        (
            ["--acknowledge-session", "s1", "--acknowledge-breaking", "115"],
            {
                "acknowledge_breaking": [115],
                "acknowledge_sessions": ["s1"],
                "session_policy": "proceed",
            },
        ),
    ],
)
def test_deployment_retry_sends_the_choice(tmp_path, flags, payload):
    client = Client(_project(tmp_path))
    arguments = cloud._build_parser().parse_args(
        [
            "deployment",
            "retry",
            "dep_old",
            "--application",
            "session-agent",
            "--idempotency-key",
            "one-submission",
            *flags,
        ]
    )
    cloud._deployment(arguments, client=client)
    expected = {"idempotency_key": "one-submission"}
    if payload is not None:
        expected["payload"] = payload
    assert client.requests[-1] == (
        "POST",
        "/v1/applications/session-agent/deployments/dep_old/retry",
        expected,
    )


class _RollbackClient:
    def __init__(self):
        self.requests = []

    def request(self, method, path, **kwargs):
        self.requests.append((method, path, kwargs))
        if path == "/v1/applications":
            return {"items": [{"id": "session-agent", "name": "Session Agent", "revision": 4}]}
        if path == "/v1/applications/session-agent":
            return {"id": "session-agent", "revision": 4}
        if path.endswith("/rollback"):
            return {"id": "session-agent", "current_deployment_id": _DEPLOYMENT}
        if path.endswith("/service"):
            return {"deployment_id": "dep_newer", "status": "running", "issues": []}
        raise AssertionError((method, path))


@pytest.mark.parametrize(
    "flags,fields",
    [
        ([], {}),
        (["--session-policy", "block"], {"session_policy": "block"}),
        (
            ["--acknowledge-session", "*"],
            {"session_policy": "proceed", "acknowledge_sessions": ["*"]},
        ),
    ],
)
def test_rollback_sends_the_choice_in_its_body(flags, fields):
    client = _RollbackClient()
    arguments = cloud._build_parser().parse_args(
        ["rollback", _DEPLOYMENT, "--application", "session-agent", *flags]
    )
    cloud._rollback(arguments, client=client)
    rollback = [kwargs for _, path, kwargs in client.requests if path.endswith("/rollback")]
    assert rollback == [{"payload": {"expected_application_revision": 4, **fields}}]


# Rerunning deploy with a new choice for unchanged source


_SESSION_REFUSAL = {
    "acknowledge_breaking": [],
    "acknowledge_sessions": ["s-input"],
    "code": "unfinished_sessions",
    "detail": (
        "s-input is paused on a pending user_input bound to the serving release. Cloud waited "
        "15 minutes and they still block. Nothing was changed; the previous release keeps "
        "serving."
    ),
    "hint": (
        "Answer or finish them and retry, or accept that they may not resume: in the Cayu "
        "Cloud portal open this Release and choose Acknowledge and retry (or Retry and wait). "
        "With a cayu CLI that has --acknowledge-session (newer than 0.10.0): `cayu cloud "
        "deployment retry dep_old --application session-agent --acknowledge-session s-input`."
    ),
    "message": "1 unfinished session of the serving release may not resume on this Release.",
    "retryable": True,
}


def _waiting_preflight(**changes):
    return {
        "api_path": "/api",
        "checked_at": "2026-10-09T12:00:30+00:00",
        "counts": {"at_boundary": 0, "in_interaction": 1, "bound_to_release": 1},
        "deadline_at": "2026-10-09T12:15:00+00:00",
        "detail": None,
        "message": (
            "Waiting for 2 unfinished sessions to reach a safe boundary before replacing the "
            "serving release (1 mid-interaction, 1 paused on a pending action)."
        ),
        "policy": "wait",
        "read": "complete",
        "reason": None,
        "serving_deployment_id": "dep_serving",
        "sessions": [],
        "state": "waiting",
        "truncated": False,
        "waiting_for": "sessions",
        "waiting_since": "2026-10-09T12:00:00+00:00",
        **changes,
    }


class RefusedReleaseClient(Client):
    """The same source was deployed with the default choice; Cloud refused or holds it.

    `reads` scripts the Release's reads in turn (the last repeats); `conflict` makes every
    create conflict with the earlier submission, as another key for the same version does.
    """

    def __init__(
        self,
        project,
        *,
        current=None,
        stored_choice=None,
        status="smoke_tested",
        reads=(),
        conflict=False,
        on_promote=None,
        after_retry=None,
        serving="dep_old",
    ):
        super().__init__(project, status="smoke_tested")
        payload = project.manifest.deployment_payload(
            repository=project.repository, revision=project.revision
        )
        manifest = dict(payload["manifest"])
        runtime = manifest.pop("runtime", None)
        if runtime is not None:
            # Cloud's canonical manifest includes a web process's unset idle timeout.
            manifest["runtime"] = {
                **runtime,
                "web": {"idle_timeout_seconds": None, **runtime["web"]},
            }
        self.release = {
            "id": "dep_old",
            "version": payload["version"],
            "manifest_digest": hashlib.sha256(
                json.dumps(manifest, separators=(",", ":"), sort_keys=True).encode()
            ).hexdigest(),
            "policy_version": payload["policy_version"],
            "created_at": "2026-10-09T00:00:00Z",
            "status": status,
            "acknowledge_breaking": [],
            "session_policy": stored_choice
            or {"acknowledge_sessions": [], "mode": "wait", "wait_seconds": 900},
        }
        self.current = {**self.release, "publication_error": _SESSION_REFUSAL}
        if current is not None:
            self.current = {**self.release, **current}
        self.reads = [{**self.release, **read} for read in reads]
        self.conflict = conflict
        # What later reads show once a promote or retry is accepted.
        self.on_promote = on_promote or {}
        self.after_retry = after_retry
        self.serving = serving
        # The choice a retry or promote stored since, which later reads carry.
        self.stored = None

    def request(self, method, path, **kwargs):
        if method == "POST" and path.endswith("/deployments"):
            self.requests.append((method, path, kwargs))
            if self.conflict or "session_policy" in kwargs["payload"]:
                raise CloudApiError(
                    "api_request_rejected",
                    "Cayu Cloud API returned HTTP 409.",
                    status_code=409,
                    detail="Application already has a deployment with this version.",
                )
            return self.release.copy()
        if method == "GET" and path.endswith("/deployments"):
            self.requests.append((method, path, kwargs))
            return {"items": [self.release.copy()], "next_cursor": None}
        if method == "GET" and path.endswith("/deployments/dep_old"):
            self.requests.append((method, path, kwargs))
            if self.reads:
                read = self.reads.pop(0) if len(self.reads) > 1 else self.reads[0]
            else:
                read = self.current
            return read if self.stored is None else {**read, "session_policy": self.stored}
        if method == "POST" and path.endswith("/dep_old/retry"):
            self.requests.append((method, path, kwargs))
            payload = kwargs["payload"]
            if "session_policy" in payload:
                self.stored = _cloud_choice(payload)
            if self.after_retry is not None:
                self.reads = [{**self.release, **self.after_retry}]
            return {
                **self.release,
                "acknowledge_breaking": payload.get("acknowledge_breaking", []),
                "session_policy": self.stored or self.release["session_policy"],
                "status": "smoke_tested",
            }
        if method == "POST" and path.endswith("/dep_old/promote"):
            self.requests.append((method, path, kwargs))
            # Cloud selects the Release and stores a choice the promote carries.
            payload = kwargs["payload"]
            if "session_policy" in payload:
                self.stored = _cloud_choice(payload)
            last = self.reads[-1] if self.reads else self.current
            self.reads = [{**last, "status": "promoted", **self.on_promote}]
            return {"id": "session-agent", "revision": 2, "current_deployment_id": "dep_old"}
        if method == "GET" and path.endswith("/service"):
            self.requests.append((method, path, kwargs))
            return {"deployment_id": self.serving, "issues": [], "status": "running"}
        if method == "GET" and path == "/v1/applications/session-agent":
            self.requests.append((method, path, kwargs))
            return {"id": "session-agent", "revision": 2, "current_deployment_id": "dep_old"}
        return super().request(method, path, **kwargs)


def test_rerun_with_acknowledged_sessions_retries_the_refused_release(tmp_path):
    project = _project(tmp_path)
    client = RefusedReleaseClient(project)
    result = _deploy(client, project, tmp_path, "--acknowledge-session", "s-input")
    assert result["result"]["retry"] == {
        "deployment_id": "dep_old",
        "failure_code": "unfinished_sessions",
        "previous_deployment_id": "dep_old",
        "reason": _SESSION_REFUSAL["message"],
        "session_policy": {
            "acknowledge_sessions": ["s-input"],
            "mode": "proceed",
            "wait_seconds": 900,
        },
    }
    (retry,) = _posts(client, "/retry")
    assert retry["payload"] == {"acknowledge_sessions": ["s-input"], "session_policy": "proceed"}
    assert retry["idempotency_key"].startswith("deploy-acknowledge:")


def test_rerun_with_a_new_choice_changes_a_waiting_release(tmp_path):
    project = _project(tmp_path)
    client = RefusedReleaseClient(project, current={"session_preflight": _waiting_preflight()})
    result = _deploy(client, project, tmp_path, "--session-policy", "block")
    assert result["result"]["retry"]["reason"] == _waiting_preflight()["message"]
    assert "failure_code" not in result["result"]["retry"]
    assert _posts(client, "/retry")[0]["payload"] == {"session_policy": "block"}


_PROCEED_ALL = {"acknowledge_sessions": ["*"], "mode": "proceed", "wait_seconds": 900}


@pytest.mark.parametrize(
    "current",
    [
        # Still building: Cloud checks it later under `proceed *`.
        {"status": "image_built"},
        # Its check published by acknowledging a session `block` protects.
        {"session_preflight": _ACKNOWLEDGED_S1},
        # A check this CLI can't read is treated as the worst case.
        {"session_preflight": {"state": "reconsidering"}},
    ],
)
def test_rerun_without_waiting_fails_when_a_more_permissive_choice_stays(tmp_path, current, capsys):
    # Asked for `block`; the Release keeps `proceed *`, and this deploy left it to publish.
    project = _project(tmp_path)
    client = RefusedReleaseClient(project, stored_choice=_PROCEED_ALL, current=current)
    with pytest.raises(CloudApiError) as raised:
        _deploy(client, project, tmp_path, "--session-policy", "block")
    assert not _posts(client, "/retry")
    error = _failure_envelope(raised, capsys)
    assert error["category"] == "session_choice_not_applied"
    assert error["deployment_id"] == "dep_old"
    assert error["commands"]["status"].endswith(
        "deployment status dep_old --application session-agent"
    )
    not_applied = error["not_applied"]
    assert not_applied["session_policy"] == {
        "acknowledge_sessions": [],
        "mode": "block",
        "wait_seconds": 900,
    }
    assert not_applied["release_session_policy"] == _PROCEED_ALL
    assert not_applied["confirmed"] is True
    assert not_applied["more_permissive"] is True
    assert "did not apply this command's session choice" in error["message"]


@pytest.mark.parametrize(
    "stored,current",
    [
        # Its recorded check found nothing blocking, so `block` would have published too.
        (_PROCEED_ALL, {"session_preflight": {"state": "clear", "policy": "proceed"}}),
        # Refused: nothing was published.
        (_PROCEED_ALL, {"session_preflight": {**_BLOCKED, "policy": "proceed"}}),
        # `wait` keeps every session `block` protects; the difference is only reported.
        (None, {"status": "image_built"}),
    ],
)
def test_rerun_without_waiting_reports_a_kept_choice_that_strands_nothing(
    tmp_path, stored, current, capsys
):
    project = _project(tmp_path)
    client = RefusedReleaseClient(project, stored_choice=stored, current=current)
    result = _deploy(client, project, tmp_path, "--session-policy", "block")
    not_applied = result["result"]["not_applied"]
    assert not_applied["more_permissive"] is False
    assert not_applied["release_session_policy"] != not_applied["session_policy"]
    assert not_applied["message"] in capsys.readouterr().err


@pytest.mark.parametrize("serving,fails", [("dep_serving", True), ("dep_old", False)])
def test_promoted_release_counts_as_published_only_once_its_service_runs_it(
    tmp_path, serving, fails, capsys
):
    # Selected under `proceed *` but not yet checked: it is publishing, not published,
    # unless the Agent's service already runs it.
    project = _project(tmp_path)
    client = RefusedReleaseClient(
        project,
        stored_choice=_PROCEED_ALL,
        status="promoted",
        current={"status": "promoted"},
        serving=serving,
    )
    if not fails:
        result = _deploy(client, project, tmp_path, "--session-policy", "block")
        assert result["result"]["not_applied"]["more_permissive"] is True
        return
    with pytest.raises(CloudApiError) as raised:
        _deploy(client, project, tmp_path, "--session-policy", "block")
    assert _failure_envelope(raised, capsys)["category"] == "session_choice_not_applied"


def test_promoted_release_whose_check_passed_counts_as_published(tmp_path):
    project = _project(tmp_path)
    client = RefusedReleaseClient(
        project,
        stored_choice=_PROCEED_ALL,
        status="promoted",
        current={"status": "promoted", "session_preflight": _ACKNOWLEDGED_S1},
        serving="dep_serving",
    )
    # The list shows the passed check, so the service isn't read.
    client.release["session_preflight"] = _ACKNOWLEDGED_S1
    result = _deploy(client, project, tmp_path, "--session-policy", "block")
    assert result["result"]["not_applied"]["more_permissive"] is True
    assert not [path for _, path, _ in client.requests if path.endswith("/service")]


def test_rerun_without_waiting_reports_a_stricter_kept_choice(tmp_path, capsys):
    project = _project(tmp_path)
    block = {"acknowledge_sessions": [], "mode": "block", "wait_seconds": 900}
    client = RefusedReleaseClient(project, stored_choice=block, current={"status": "image_built"})
    result = _deploy(client, project, tmp_path, "--acknowledge-session", "*")
    not_applied = result["result"]["not_applied"]
    assert not_applied["more_permissive"] is False
    assert not_applied["release_session_policy"]["mode"] == "block"
    assert not_applied["message"] in capsys.readouterr().err


def test_rerun_with_the_release_choice_already_stored_does_not_retry(tmp_path):
    project = _project(tmp_path)
    client = RefusedReleaseClient(
        project,
        stored_choice={"acknowledge_sessions": [], "mode": "block", "wait_seconds": 900},
    )
    result = _deploy(client, project, tmp_path, "--session-policy", "block")
    assert "retry" not in result["result"]
    assert not _posts(client, "/retry")


def test_rerun_with_the_default_choice_retries_a_release_refused_under_another(tmp_path):
    # `deploy --session-policy block` was refused; rerunning with Cloud's default choice
    # keeps the legacy create (which conflicts) and retries the Release with `wait`.
    project = _project(tmp_path)
    block = {"acknowledge_sessions": [], "mode": "block", "wait_seconds": 900}
    client = RefusedReleaseClient(project, stored_choice=block, conflict=True)
    result = _deploy(client, project, tmp_path, "--session-policy", "wait")
    (create,) = _posts(client, "/deployments")
    assert create["idempotency_key"] == _legacy_source_key(project)
    assert create["payload"] == project.manifest.deployment_payload(
        repository=project.repository, revision=project.revision
    )
    (retry,) = _posts(client, "/retry")
    assert retry["payload"] == {"session_policy": "wait"}
    assert result["result"]["retry"]["session_policy"] == {
        "acknowledge_sessions": [],
        "mode": "wait",
        "wait_seconds": 900,
    }
    assert "not_applied" not in result["result"]


def test_rerun_with_an_acknowledgement_and_the_default_choice_carries_both(tmp_path):
    project = _project(tmp_path)
    block = {"acknowledge_sessions": [], "mode": "block", "wait_seconds": 900}
    client = RefusedReleaseClient(project, stored_choice=block, conflict=True)
    _deploy(client, project, tmp_path, "--acknowledge-breaking", "115", "--session-policy", "wait")
    assert _posts(client, "/retry")[0]["payload"] == {
        "acknowledge_breaking": [115],
        "session_policy": "wait",
    }


def test_replayed_release_gets_the_choice_given_again(tmp_path):
    # The create replays the Release (same key), but a retry changed its choice since.
    project = _project(tmp_path)
    block = {"acknowledge_sessions": [], "mode": "block", "wait_seconds": 900}
    client = RefusedReleaseClient(project, stored_choice=block)
    result = _deploy(client, project, tmp_path, "--session-policy", "wait")
    assert _posts(client, "/retry")[0]["payload"] == {"session_policy": "wait"}
    assert result["result"]["retry"]["failure_code"] == "unfinished_sessions"


def test_rerun_found_building_carries_its_choice_on_the_promote(tmp_path):
    project = _project(tmp_path)
    client = RefusedReleaseClient(
        project,
        status="image_built",
        reads=[{"status": "image_built"}, {"status": "image_built"}, {"status": "smoke_tested"}],
    )
    result = _deploy(client, project, tmp_path, "--session-policy", "block", wait=True)
    assert not _posts(client, "/retry")
    (promote,) = _posts(client, "/promote")
    assert promote["payload"] == {"expected_application_revision": 1, "session_policy": "block"}
    assert result["result"]["deployment"]["session_policy"]["mode"] == "block"
    assert "not_applied" not in result["result"]


def test_rerun_found_building_retries_once_cloud_holds_it(tmp_path):
    project = _project(tmp_path)
    held = {"status": "smoke_tested", "session_preflight": _waiting_preflight()}
    client = RefusedReleaseClient(
        project,
        status="image_built",
        reads=[{"status": "image_built"}, held, held, {"status": "promoted"}],
    )
    result = _deploy(client, project, tmp_path, "--acknowledge-session", "*", wait=True)
    (retry,) = _posts(client, "/retry")
    assert retry["payload"] == {"acknowledge_sessions": ["*"], "session_policy": "proceed"}
    assert result["result"]["retry"]["reason"] == _waiting_preflight()["message"]
    assert not _posts(client, "/promote")


def test_rerun_found_published_reports_the_choice_as_not_applied(tmp_path, capsys):
    project = _project(tmp_path)
    client = RefusedReleaseClient(
        project,
        status="promoted",
        reads=[{"status": "promoted", "session_preflight": {"state": "clear"}}],
    )
    result = _deploy(client, project, tmp_path, "--session-policy", "block", wait=True)
    assert not _posts(client, "/retry")
    assert not _posts(client, "/promote")
    assert result["result"]["not_applied"]["session_policy"]["mode"] == "block"
    assert "keeps its earlier `wait` choice" in capsys.readouterr().err


def test_rerun_retrying_a_held_release_does_not_claim_the_acknowledgement(tmp_path, capsys):
    # Cloud's retry of a publication waiting for sessions only changes its choice.
    project = _project(tmp_path)
    client = RefusedReleaseClient(project, current={"session_preflight": _waiting_preflight()})
    result = _deploy(
        client, project, tmp_path, "--acknowledge-breaking", "115", "--session-policy", "block"
    )
    assert _posts(client, "/retry")[0]["payload"] == {"session_policy": "block"}
    retry = result["result"]["retry"]
    assert "acknowledge_breaking" not in retry
    assert retry["acknowledge_breaking_not_applied"] == [115]
    not_applied = result["result"]["not_applied"]
    assert not_applied["acknowledge_breaking"] == [115]
    assert "session_policy" not in not_applied
    assert "--acknowledge-breaking 115" in capsys.readouterr().err


def test_rerun_retrying_a_refused_release_carries_the_acknowledgement(tmp_path):
    project = _project(tmp_path)
    client = RefusedReleaseClient(project)
    result = _deploy(
        client, project, tmp_path, "--acknowledge-breaking", "115", "--session-policy", "block"
    )
    assert _posts(client, "/retry")[0]["payload"] == {
        "acknowledge_breaking": [115],
        "session_policy": "block",
    }
    retry = result["result"]["retry"]
    assert retry["acknowledge_breaking"] == [115]
    assert "acknowledge_breaking_not_applied" not in retry


@pytest.mark.parametrize(
    "response,not_applied",
    [
        # Cloud's retry of a held publication changed the choice only.
        (
            {
                "acknowledge_breaking": [],
                "session_policy": _cloud_choice({"session_policy": "block"}),
            },
            {"acknowledge_breaking": [115]},
        ),
        # A refused publication took both.
        (
            {
                "acknowledge_breaking": [115],
                "session_policy": _cloud_choice({"session_policy": "block"}),
            },
            None,
        ),
        # An attempt already running took neither, and keeps a stricter choice.
        (
            {
                "acknowledge_breaking": [],
                "session_policy": _cloud_choice({"session_policy": "block"}) | {"wait_seconds": 60},
            },
            {"acknowledge_breaking": [115], "session_policy": "block"},
        ),
    ],
)
def test_deployment_retry_reports_what_cloud_did_not_apply(response, not_applied, capsys):
    class RetryClient:
        def request(self, method, path, **kwargs):
            if path == "/v1/applications":
                return {"items": [{"id": "session-agent", "name": "Session Agent"}]}
            return {"id": "dep_old", "status": "smoke_tested", **response}

    arguments = cloud._build_parser().parse_args(
        [
            "deployment",
            "retry",
            "dep_old",
            "--application",
            "session-agent",
            "--acknowledge-breaking",
            "115",
            "--session-policy",
            "block",
        ]
    )
    output = cloud._deployment(arguments, client=RetryClient())
    if not_applied is None:
        assert "not_applied" not in output
        assert capsys.readouterr().err == ""
        return
    reported = output["not_applied"]
    assert reported.get("acknowledge_breaking") == not_applied.get("acknowledge_breaking")
    assert ("session_policy" in reported) == ("session_policy" in not_applied)
    assert reported["message"] in capsys.readouterr().err


def test_deployment_retry_fails_when_a_running_attempt_keeps_a_more_permissive_choice(capsys):
    class RetryClient:
        def request(self, method, path, **kwargs):
            if path == "/v1/applications":
                return {"items": [{"id": "session-agent", "name": "Session Agent"}]}
            return {
                "id": "dep_old",
                "session_policy": _cloud_choice({"acknowledge_sessions": ["*"]}),
                "status": "accepted",
            }

    arguments = cloud._build_parser().parse_args(
        [
            "deployment",
            "retry",
            "dep_old",
            "--application",
            "session-agent",
            "--session-policy",
            "block",
        ]
    )
    with pytest.raises(CloudApiError) as raised:
        cloud._deployment(arguments, client=RetryClient())
    error = _failure_envelope(raised, capsys)
    assert error["category"] == "session_choice_not_applied"
    assert error["not_applied"]["release_session_policy"]["acknowledge_sessions"] == ["*"]


def _promoted_rerun(tmp_path, *flags, on_promote):
    # The rerun finds the Release building under `proceed *`; Cloud's finalize may check
    # its sessions before or after the promote stores this deploy's choice.
    project = _project(tmp_path, web=True)
    client = RefusedReleaseClient(
        project,
        stored_choice=_cloud_choice({"acknowledge_sessions": ["*"]}),
        status="image_built",
        reads=[{"status": "image_built"}, {"status": "image_built"}, {"status": "smoke_tested"}],
        on_promote=on_promote,
    )
    return client, lambda: _deploy(client, project, tmp_path, *flags, wait=True)


def test_promoted_choice_is_not_applied_when_the_check_passed_under_the_earlier_one(
    tmp_path, capsys
):
    client, deploy = _promoted_rerun(
        tmp_path, "--session-policy", "block", on_promote={"session_preflight": _ACKNOWLEDGED_S1}
    )
    with pytest.raises(CloudApiError) as raised:
        deploy()
    assert _posts(client, "/promote")[0]["payload"]["session_policy"] == "block"
    error = _failure_envelope(raised, capsys)
    assert error["category"] == "session_choice_not_applied"
    not_applied = error["not_applied"]
    # Cloud stored `block`, but its check had already passed under `proceed`.
    assert not_applied["release_session_policy"]["mode"] == "block"
    assert not_applied["checked_session_policy"] == {
        "acknowledge_sessions": ["s1"],
        "mode": "proceed",
    }
    assert "passed its session check under the earlier `proceed` choice" in error["message"]


@pytest.mark.parametrize(
    "flags,preflight",
    [
        # Nothing blocked, so any choice publishes.
        (["--session-policy", "block"], {"state": "clear", "policy": "proceed"}),
        # The acknowledged sessions are ones this deploy acknowledges too.
        (
            ["--acknowledge-session", "s1", "--acknowledge-session", "s2"],
            _ACKNOWLEDGED_S1,
        ),
    ],
)
def test_promoted_choice_agrees_with_the_recorded_check(tmp_path, flags, preflight):
    _client, deploy = _promoted_rerun(tmp_path, *flags, on_promote={"session_preflight": preflight})
    assert "not_applied" not in deploy()["result"]


def test_promoted_choice_without_a_recorded_check_is_unconfirmed(tmp_path, capsys):
    _client, deploy = _promoted_rerun(tmp_path, "--session-policy", "block", on_promote={})
    not_applied = deploy()["result"]["not_applied"]
    assert not_applied["confirmed"] is False
    assert not_applied["more_permissive"] is False
    assert "isn't confirmed" in capsys.readouterr().err


@pytest.mark.parametrize(
    "kept,requested,more",
    [
        # `wait` and `block` never publish over a blocking session.
        ({"mode": "wait"}, {"mode": "block"}, False),
        ({"mode": "block"}, {"mode": "wait"}, False),
        ({"mode": "wait", "wait_seconds": 3600}, {"mode": "wait"}, False),
        ({"mode": "proceed", "acknowledge_sessions": ["s1"]}, {"mode": "wait"}, True),
        (
            {"mode": "proceed", "acknowledge_sessions": ["s1"]},
            {"mode": "proceed", "acknowledge_sessions": ["s1", "s2"]},
            False,
        ),
        (
            {"mode": "proceed", "acknowledge_sessions": ["s1", "s3"]},
            {"mode": "proceed", "acknowledge_sessions": ["s1", "s2"]},
            True,
        ),
        (
            {"mode": "proceed", "acknowledge_sessions": ["*"]},
            {"mode": "proceed", "acknowledge_sessions": ["s1"]},
            True,
        ),
        (
            {"mode": "proceed", "acknowledge_sessions": ["s9"]},
            {"mode": "proceed", "acknowledge_sessions": ["*"]},
            False,
        ),
        # Nothing is more permissive than '*'.
        ({"mode": "later"}, {"mode": "proceed", "acknowledge_sessions": ["*"]}, False),
        ({"mode": "later"}, {"mode": "block"}, True),
    ],
)
def test_session_choices_rank_by_what_they_let_publish(kept, requested, more):
    requested = {"acknowledge_sessions": [], "wait_seconds": 900, **requested}
    assert cloud._more_permissive_session_choice(kept, requested) is more


@pytest.mark.parametrize(
    "preflight,more",
    [
        (None, True),
        # Held: Cloud decides later under the kept choice.
        (_waiting_preflight(policy="proceed"), True),
        ({"state": "clear"}, False),
        (_BLOCKED, False),
        (_ACKNOWLEDGED_S1, True),
        ({**_ACKNOWLEDGED_S1, "truncated": True}, True),
        ({"state": "reconsidering"}, True),
    ],
)
def test_a_recorded_check_decides_whether_the_kept_choice_strands_sessions(preflight, more):
    release = {"session_preflight": preflight}
    kept = {"acknowledge_sessions": ["*"], "mode": "proceed", "wait_seconds": 900}
    block = {"acknowledge_sessions": [], "mode": "block", "wait_seconds": 900}
    assert cloud._more_permissive_publication(release, kept, block) is more


def test_rerun_retries_a_refusal_that_lands_after_the_built_check(tmp_path):
    # Built with nothing to change yet; Cloud then refuses it under the earlier choice.
    project = _project(tmp_path)
    refused = {"status": "smoke_tested", "publication_error": _SESSION_REFUSAL}
    client = RefusedReleaseClient(
        project,
        status="image_built",
        reads=[
            {"status": "image_built"},
            {"status": "smoke_tested"},
            {"status": "smoke_tested"},
            refused,
        ],
        after_retry={"status": "promoted"},
    )
    result = _deploy(client, project, tmp_path, "--acknowledge-session", "s-input", wait=True)
    (retry,) = _posts(client, "/retry")
    assert retry["payload"] == {"acknowledge_sessions": ["s-input"], "session_policy": "proceed"}
    assert result["result"]["retry"]["failure_code"] == "unfinished_sessions"
    assert result["result"]["deployment"]["status"] == "promoted"
    assert "not_applied" not in result["result"]


# Refusals


class _PublicationClient:
    def __init__(self, publication_error, *, status="promoted", **release):
        self.publication_error = publication_error
        self.status = status
        # The refused Release's stored choice and preflight.
        self.release = {"session_preflight": _BLOCKED, **release}
        self.requests = []

    def request(self, method, path, **kwargs):
        self.requests.append((method, path, kwargs))
        if path == "/v1/applications":
            return {"items": [{"id": "ready-agent", "name": "Ready Agent", "revision": 3}]}
        if path == "/v1/applications/ready-agent":
            return {"id": "ready-agent", "revision": 3}
        if path.endswith("/rollback"):
            return {"id": "ready-agent", "current_deployment_id": _DEPLOYMENT}
        if path.endswith("/service"):
            return {"deployment_id": "dep_newer", "status": "running", "issues": []}
        assert (method, path) == ("GET", f"/v1/applications/ready-agent/deployments/{_DEPLOYMENT}")
        return {
            "id": _DEPLOYMENT,
            "status": self.status,
            "publication_error": self.publication_error,
            **self.release,
        }


def _failure_envelope(raised, capsys):
    assert cloud._cloud_failure(raised.value) == 2
    return json.loads(capsys.readouterr().out)["error"]


def _wait_for_service(client, **kwargs):
    return cloud._wait_for_service(
        client,
        application_id="ready-agent",
        initial=None,
        expected_deployment_id=_DEPLOYMENT,
        poll_seconds=0.01,
        wait_seconds=600,
        sleep=lambda _: None,
        monotonic=lambda: 0,
        **kwargs,
    )


_BASE = (
    f"cayu cloud --context /private/cloud.json deployment %s {_DEPLOYMENT} "
    "--application ready-agent"
)


def test_unfinished_sessions_refusal_prints_the_acknowledging_retry(capsys):
    refusal = {**_SESSION_REFUSAL, "acknowledge_sessions": ["s-input", "s-busy"]}
    with pytest.raises(CloudApiError) as raised:
        _wait_for_service(
            _PublicationClient(refusal), recovery_arguments=("--context", "/private/cloud.json")
        )
    error = _failure_envelope(raised, capsys)
    assert error["category"] == "service_publication_failed"
    assert error["code"] == "unfinished_sessions"
    assert error["acknowledge_sessions"] == ["s-busy", "s-input"]
    assert error["publication_error"]["acknowledge_sessions"] == ["s-busy", "s-input"]
    retry = _BASE % "retry" + " --acknowledge-session s-busy --acknowledge-session s-input"
    wait = _BASE % "retry" + " --session-policy wait"
    assert error["commands"] == {
        "retry": retry,
        "retry_wait": wait,
        "status": _BASE % "status",
        "timeline": _BASE % "timeline",
    }
    message = error["message"]
    assert message.startswith(f"{refusal['message']} {refusal['detail']} ")
    assert f"accept that they may not resume on this release with `{retry}`" in message
    assert f"wait for them again with `{wait}`" in message
    # Cloud's hint, written for CLIs without the flag, is replaced.
    assert "portal" not in message


def test_sessions_unreadable_refusal_acknowledges_every_session(capsys):
    refusal = {
        **_SESSION_REFUSAL,
        "acknowledge_sessions": ["*"],
        "code": "sessions_unreadable",
        "detail": "The Agent did not answer within 10 seconds. Nothing was changed.",
        "message": "Cloud could not read the serving release's unfinished sessions.",
    }
    with pytest.raises(CloudApiError) as raised:
        _wait_for_service(_PublicationClient(refusal))
    error = _failure_envelope(raised, capsys)
    base = f"cayu cloud deployment %s {_DEPLOYMENT} --application ready-agent"
    assert error["code"] == "sessions_unreadable"
    assert error["commands"]["retry"] == base % "retry" + " --acknowledge-session '*'"
    assert error["commands"]["retry_wait"] == base % "retry" + " --session-policy wait"
    assert "accept that every unfinished session may not resume" in error["message"]
    assert "wait for Cloud to read them" in error["message"]


@pytest.mark.parametrize(
    "sessions",
    [None, [], "s-input", ["s-input", ""], ["s\x00"], [f"s{index}" for index in range(201)]],
)
def test_refusal_without_valid_sessions_keeps_cloud_hint_and_the_wait_alternative(sessions, capsys):
    refusal = {
        key: value for key, value in _SESSION_REFUSAL.items() if key != "acknowledge_sessions"
    }
    if sessions is not None:
        refusal["acknowledge_sessions"] = sessions
    with pytest.raises(CloudApiError) as raised:
        _wait_for_service(_PublicationClient(refusal))
    error = _failure_envelope(raised, capsys)
    assert error["code"] == "unfinished_sessions"
    assert "retry" not in error["commands"]
    assert "acknowledge_sessions" not in error
    assert "acknowledge_sessions" not in error["publication_error"]
    assert error["commands"]["retry_wait"].endswith("--session-policy wait")
    assert f"{_SESSION_REFUSAL['hint']} Or wait for them again with `" in error["message"]


def test_refusal_retry_keeps_the_sessions_the_release_already_acknowledged(capsys):
    # The Release proceeds past s-a; s-b became busy since. A retry replaces the choice,
    # so naming only Cloud's s-b would leave s-a blocking it next time.
    refusal = {**_SESSION_REFUSAL, "acknowledge_sessions": ["s-b"]}
    stored = {"acknowledge_sessions": ["s-c", "s-a"], "mode": "proceed", "wait_seconds": 900}
    with pytest.raises(CloudApiError) as raised:
        _wait_for_service(_PublicationClient(refusal, session_policy=stored))
    error = _failure_envelope(raised, capsys)
    base = f"cayu cloud deployment retry {_DEPLOYMENT} --application ready-agent"
    assert error["commands"]["retry"] == (
        base + " --acknowledge-session s-a --acknowledge-session s-b --acknowledge-session s-c"
    )
    assert error["acknowledge_sessions"] == ["s-a", "s-b", "s-c"]
    # Cloud's own list is reported as Cloud sent it.
    assert error["publication_error"]["acknowledge_sessions"] == ["s-b"]


def test_refusal_retry_ignores_a_stored_choice_that_is_not_proceed(capsys):
    refusal = {**_SESSION_REFUSAL, "acknowledge_sessions": ["s-b"]}
    stored = {"acknowledge_sessions": [], "mode": "block", "wait_seconds": 900}
    with pytest.raises(CloudApiError) as raised:
        _wait_for_service(_PublicationClient(refusal, session_policy=stored))
    error = _failure_envelope(raised, capsys)
    assert error["commands"]["retry"].endswith(
        "--application ready-agent --acknowledge-session s-b"
    )


@pytest.mark.parametrize(
    "release",
    [
        # Cloud lists at most 100 sessions; more block than it names.
        {"session_preflight": {**_BLOCKED, "truncated": True}},
        # Nothing confirms Cloud's list is complete.
        {"session_preflight": None},
        {"session_preflight": {"read": "complete", "state": "blocked"}},
        # The merged list exceeds what a retry can name.
        {
            "session_policy": {
                "acknowledge_sessions": [f"s{index:03d}" for index in range(200)],
                "mode": "proceed",
                "wait_seconds": 900,
            }
        },
        # A stored proceed choice that can't be read can't be kept.
        {"session_policy": {"acknowledge_sessions": [""], "mode": "proceed", "wait_seconds": 900}},
    ],
)
def test_refusal_without_a_retry_that_can_pass_keeps_cloud_hint(release, capsys):
    with pytest.raises(CloudApiError) as raised:
        _wait_for_service(_PublicationClient(_SESSION_REFUSAL, **release))
    error = _failure_envelope(raised, capsys)
    assert error["code"] == "unfinished_sessions"
    assert "retry" not in error["commands"]
    assert "acknowledge_sessions" not in error
    assert error["commands"]["retry_wait"].endswith("--session-policy wait")
    assert f"{_SESSION_REFUSAL['hint']} Or wait for them again with `" in error["message"]


def test_unreadable_refusal_acknowledges_everyone_even_when_truncated(capsys):
    refusal = {**_SESSION_REFUSAL, "acknowledge_sessions": ["*"], "code": "sessions_unreadable"}
    release = {"session_preflight": {**_BLOCKED, "read": "partial", "truncated": True}}
    with pytest.raises(CloudApiError) as raised:
        _wait_for_service(_PublicationClient(refusal, **release))
    error = _failure_envelope(raised, capsys)
    assert error["commands"]["retry"].endswith("--acknowledge-session '*'")


@pytest.mark.parametrize(
    "stored,flags",
    [
        (
            {"acknowledge_sessions": [], "mode": "wait", "wait_seconds": 1800},
            " --session-wait-seconds 1800",
        ),
        ({"acknowledge_sessions": [], "mode": "wait", "wait_seconds": 120}, ""),
        ({"acknowledge_sessions": [], "mode": "wait", "wait_seconds": 9000}, ""),
        (None, ""),
    ],
)
def test_refusal_wait_alternative_keeps_a_longer_stored_wait(stored, flags, capsys):
    with pytest.raises(CloudApiError) as raised:
        _wait_for_service(_PublicationClient(_SESSION_REFUSAL, session_policy=stored))
    error = _failure_envelope(raised, capsys)
    wait = (
        f"cayu cloud deployment retry {_DEPLOYMENT} --application ready-agent --session-policy wait"
    )
    assert error["commands"]["retry_wait"] == wait + flags
    assert f"`{wait + flags}`" in error["message"]


def test_deployment_wait_reports_a_session_refusal(capsys):
    client = _PublicationClient(_SESSION_REFUSAL, status="smoke_tested")
    arguments = cloud._build_parser().parse_args(
        ["deployment", "wait", _DEPLOYMENT, "--application", "ready-agent"]
    )
    with pytest.raises(CloudApiError) as raised:
        cloud._deployment(arguments, client=client)
    error = _failure_envelope(raised, capsys)
    assert error["code"] == "unfinished_sessions"
    assert error["commands"]["retry"].endswith("--acknowledge-session s-input")


def test_rollback_wait_reports_a_session_refusal(capsys):
    client = _PublicationClient(_SESSION_REFUSAL)
    arguments = cloud._build_parser().parse_args(
        [
            "rollback",
            _DEPLOYMENT,
            "--application",
            "ready-agent",
            "--wait",
            "--session-policy",
            "block",
        ]
    )
    with pytest.raises(CloudApiError) as raised:
        cloud._rollback(arguments, client=client)
    error = _failure_envelope(raised, capsys)
    assert error["code"] == "unfinished_sessions"
    assert error["commands"]["retry"].endswith("--acknowledge-session s-input")


class _BuiltClient(Client):
    """A deploy whose Release Cloud built and then refused to publish before selecting it."""

    def __init__(self, project, deployment):
        super().__init__(project)
        self.deployment = deployment

    def request(self, method, path, **kwargs):
        if method == "GET" and path.endswith("/deployments/dep_old"):
            self.requests.append((method, path, kwargs))
            return self.deployment
        return super().request(method, path, **kwargs)


def test_deploy_reports_a_refusal_before_promoting(tmp_path, capsys):
    project = _project(tmp_path)
    client = _BuiltClient(
        project,
        {"id": "dep_old", "status": "smoke_tested", "publication_error": _SESSION_REFUSAL},
    )
    with pytest.raises(CloudApiError) as raised:
        cloud._deploy(
            SimpleNamespace(
                application=None,
                no_wait=False,
                no_promote=False,
                poll_seconds=1.0,
                wait_seconds=60.0,
            ),
            client=client,
            project=project,
            recorder=EvidenceRecorder(tmp_path / "evidence"),
            sleep=lambda _: None,
            monotonic=lambda: 0,
        )
    error = _failure_envelope(raised, capsys)
    assert error["code"] == "unfinished_sessions"
    assert not _posts(client, "/promote")


# Waiting


class _Clock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class _SequenceClient:
    """Answers deployment reads from a script of responses, then repeats the last."""

    def __init__(self, deployments, *, service=None, selected=_DEPLOYMENT):
        self.deployments = list(deployments)
        self.service = service
        self.selected = selected
        self.reads = 0

    def request(self, method, path, **kwargs):
        assert method == "GET"
        if path.endswith("/service"):
            return self.service(self) if callable(self.service) else self.service
        if path == "/v1/applications/ready-agent":
            return {"current_deployment_id": self.selected, "id": "ready-agent", "revision": 4}
        self.reads += 1
        if len(self.deployments) > 1:
            return self.deployments.pop(0)
        return self.deployments[0]


def _held(preflight=None, *, status="smoke_tested"):
    return {
        "id": _DEPLOYMENT,
        "status": status,
        "session_preflight": preflight or _waiting_preflight(),
    }


def _wait_for_deployment(client, clock, *, wait_seconds=10.0):
    return cloud._wait_for_deployment(
        client,
        application_id="ready-agent",
        deployment_id=_DEPLOYMENT,
        poll_seconds=5.0,
        wait_seconds=wait_seconds,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )


def test_wait_keeps_waiting_while_cloud_holds_the_release_for_sessions(capsys):
    clock = _Clock()
    # Each check is 30 seconds later on Cloud's clock, with the same deadline.
    checks = [
        _held(_waiting_preflight(checked_at=f"2026-10-09T12:{minute:02d}:30+00:00"))
        for minute in range(0, 14)
    ]
    published = {"id": _DEPLOYMENT, "status": "promoted", "session_preflight": {"state": "clear"}}
    client = _SequenceClient([*checks, published])
    result = _wait_for_deployment(client, clock)
    assert result["status"] == "promoted"
    # Far beyond the command's own 10 seconds.
    assert clock.now == 5.0 * len(checks)
    stderr = capsys.readouterr().err
    # Reported once, not on every unchanged check.
    assert stderr.count("Waiting for 2 unfinished sessions") == 1
    assert "Cayu Cloud decides by 2026-10-09T12:15:00+00:00." in stderr


def test_wait_stops_at_the_preflight_deadline_plus_its_own_wait(capsys):
    clock = _Clock()
    # Fourteen and a half minutes remain on Cloud's clock; Cloud then stops answering.
    client = _SequenceClient([_held()])
    with pytest.raises(CloudApiError) as raised:
        _wait_for_deployment(client, clock, wait_seconds=10.0)
    assert raised.value.category == "deployment_still_running"
    assert 870.0 + 10.0 <= clock.now < 870.0 + 10.0 + 5.0
    error = _failure_envelope(raised, capsys)
    assert error["last_issue"].startswith("Waiting for 2 unfinished sessions")
    assert "Last report: Waiting for 2 unfinished sessions" in error["message"]


@pytest.mark.parametrize(
    "changes,limit",
    [
        # Bounded by Cloud's longest wait, whatever deadline is reported.
        ({"deadline_at": "2027-01-01T00:00:00+00:00"}, 3600.0),
        # A deadline already past adds nothing to wait for.
        ({"deadline_at": "2026-10-09T11:00:00+00:00"}, 0.0),
    ],
)
def test_wait_bounds_the_reported_deadline(changes, limit):
    clock = _Clock()
    client = _SequenceClient([_held(_waiting_preflight(**changes))])
    with pytest.raises(CloudApiError):
        _wait_for_deployment(client, clock, wait_seconds=10.0)
    assert limit + 10.0 <= clock.now < limit + 10.0 + 5.0


@pytest.mark.parametrize(
    "changes",
    [
        {"deadline_at": None},
        {"deadline_at": "tomorrow"},
        {"deadline_at": "2026-10-09T12:15:00"},
        {"deadline_at": 1760000000},
    ],
)
def test_wait_without_a_usable_deadline_keeps_its_own(changes, capsys):
    clock = _Clock()
    client = _SequenceClient([_held(_waiting_preflight(**changes))])
    with pytest.raises(CloudApiError):
        _wait_for_deployment(client, clock, wait_seconds=10.0)
    assert clock.now == 10.0
    stderr = capsys.readouterr().err
    assert "Waiting for 2 unfinished sessions" in stderr
    assert "decides by" not in stderr


def test_wait_tolerates_unknown_fields_and_values(capsys):
    clock = _Clock()
    preflight = _waiting_preflight(
        waiting_for="Some Thing!",
        message=None,
        future_field={"anything": True},
        sessions="unexpected",
        counts=None,
    )
    published = {"id": _DEPLOYMENT, "status": "promoted"}
    client = _SequenceClient([_held(preflight), published])
    assert _wait_for_deployment(client, clock)["status"] == "promoted"
    assert "waiting for the serving release's unfinished sessions" in capsys.readouterr().err


@pytest.mark.parametrize("state", ["clear", "blocked", "acknowledged", "unsupported", None])
def test_a_preflight_that_is_not_waiting_does_not_hold_the_wait(state, capsys):
    clock = _Clock()
    client = _SequenceClient([_held(_waiting_preflight(state=state))])
    assert _wait_for_deployment(client, clock)["status"] == "smoke_tested"
    assert clock.now == 0
    assert capsys.readouterr().err == ""


def test_service_wait_follows_a_held_rollback_until_the_service_runs_it(capsys):
    clock = _Clock()

    def service(client):
        # The selected release replaces the serving one once its sessions settle.
        if client.reads >= 20:
            return {"deployment_id": _DEPLOYMENT, "status": "running", "issues": []}
        return {"deployment_id": "dep_serving", "status": "running", "issues": []}

    client = _SequenceClient(
        [_held(_waiting_preflight(waiting_for="agent_wake"), status="promoted")],
        service=service,
    )
    result = cloud._wait_for_service(
        client,
        application_id="ready-agent",
        initial=None,
        expected_deployment_id=_DEPLOYMENT,
        poll_seconds=5.0,
        wait_seconds=10.0,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )
    assert result["deployment_id"] == _DEPLOYMENT
    assert clock.now > 10.0
    assert "Waiting for 2 unfinished sessions" in capsys.readouterr().err


def test_service_wait_timeout_reports_the_hold(capsys):
    clock = _Clock()
    client = _SequenceClient(
        [_held(status="promoted")],
        service={"deployment_id": "dep_serving", "status": "running", "issues": []},
    )
    with pytest.raises(CloudApiError) as raised:
        cloud._wait_for_service(
            client,
            application_id="ready-agent",
            initial=None,
            expected_deployment_id=_DEPLOYMENT,
            poll_seconds=5.0,
            wait_seconds=10.0,
            sleep=clock.sleep,
            monotonic=clock.monotonic,
        )
    assert 880.0 <= clock.now < 885.0
    error = _failure_envelope(raised, capsys)
    assert error["category"] == "service_still_starting"
    assert error["last_issue"].startswith("Waiting for 2 unfinished sessions")


# A held Release that another one superseded


def _superseded(*, status="smoke_tested"):
    return _held(
        _waiting_preflight(
            state="superseded",
            message=(
                "Release dep_newer was selected while this one waited for unfinished sessions, "
                "so this one was not published."
            ),
        ),
        status=status,
    )


def test_wait_stops_at_a_superseded_release(capsys):
    clock = _Clock()
    client = _SequenceClient([_held(), _superseded()])
    with pytest.raises(CloudApiError) as raised:
        _wait_for_deployment(client, clock)
    assert clock.now == 5.0
    error = _failure_envelope(raised, capsys)
    assert error["category"] == "release_superseded"
    assert error["message"].startswith("Release dep_newer was selected while this one waited")
    assert "change `version` in cayu-cloud.toml" in error["message"]
    base = f"cayu cloud deployment %s {_DEPLOYMENT} --application ready-agent"
    assert error["commands"] == {"status": base % "status", "timeline": base % "timeline"}
    assert error["deployment_id"] == _DEPLOYMENT
    assert error["status"] == "smoke_tested"


def test_deployment_wait_does_not_report_a_superseded_release_ready(capsys):
    arguments = cloud._build_parser().parse_args(
        ["deployment", "wait", _DEPLOYMENT, "--application", "ready-agent"]
    )

    class Client(_SequenceClient):
        def request(self, method, path, **kwargs):
            if path == "/v1/applications":
                return {"items": [{"id": "ready-agent", "name": "Ready Agent"}]}
            return super().request(method, path, **kwargs)

    with pytest.raises(CloudApiError) as raised:
        cloud._deployment(arguments, client=Client([_superseded()]))
    assert _failure_envelope(raised, capsys)["category"] == "release_superseded"


def test_promote_race_stops_at_a_superseded_release(capsys):
    class Client(_SequenceClient):
        def request(self, method, path, **kwargs):
            if method == "POST":
                raise CloudApiError(
                    "api_request_rejected",
                    "Cayu Cloud API returned HTTP 409.",
                    status_code=409,
                    detail="Application changed after the expected revision.",
                )
            return super().request(method, path, **kwargs)

    clock = _Clock()
    with pytest.raises(CloudApiError) as raised:
        cloud._promote_release(
            Client([_superseded()]),
            application={"id": "ready-agent", "revision": 3},
            deployment_id=_DEPLOYMENT,
            wait=True,
            poll_seconds=5.0,
            recovery_arguments=(),
            wait_seconds=600.0,
            sleep=clock.sleep,
            monotonic=clock.monotonic,
        )
    # Reported at once, not after polling for the whole wait as a promotion conflict.
    assert clock.now == 0
    assert _failure_envelope(raised, capsys)["category"] == "release_superseded"


def test_a_promoted_release_with_an_old_superseded_preflight_is_ready():
    # Still the Agent's selected Release, for example after a rollback.
    clock = _Clock()
    client = _SequenceClient([_superseded(status="promoted")], selected=_DEPLOYMENT)
    assert _wait_for_deployment(client, clock)["status"] == "promoted"


def test_wait_stops_at_a_promoted_release_superseded_by_another_selection(capsys):
    # The CLI's promote selected it before finalize held it; another was selected since.
    clock = _Clock()
    client = _SequenceClient([_superseded(status="promoted")], selected="dep_newer")
    with pytest.raises(CloudApiError) as raised:
        _wait_for_deployment(client, clock)
    error = _failure_envelope(raised, capsys)
    assert error["category"] == "release_superseded"
    assert error["status"] == "promoted"
    assert "select it again with `cayu cloud rollback`" in error["message"]
    assert error["commands"]["rollback"] == (
        f"cayu cloud rollback {_DEPLOYMENT} --application ready-agent --wait"
    )


@pytest.mark.parametrize("selected,superseded", [("dep_newer", True), (_DEPLOYMENT, False)])
def test_service_wait_stops_at_a_superseded_release(selected, superseded, capsys):
    clock = _Clock()

    def service(client):
        if client.reads >= 3:
            return {"deployment_id": _DEPLOYMENT, "status": "running", "issues": []}
        return {"deployment_id": "dep_serving", "status": "running", "issues": []}

    client = _SequenceClient([_superseded(status="promoted")], service=service, selected=selected)

    def wait():
        return cloud._wait_for_service(
            client,
            application_id="ready-agent",
            initial=None,
            expected_deployment_id=_DEPLOYMENT,
            poll_seconds=5.0,
            wait_seconds=60.0,
            sleep=clock.sleep,
            monotonic=clock.monotonic,
        )

    if not superseded:
        assert wait()["deployment_id"] == _DEPLOYMENT
        return
    with pytest.raises(CloudApiError) as raised:
        wait()
    assert clock.now == 0
    assert _failure_envelope(raised, capsys)["category"] == "release_superseded"
