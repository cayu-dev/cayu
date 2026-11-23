"""`cayu cloud deploy OWNER/NAME`: Cayu Cloud builds the Agent from a GitHub repository.

Cloud reads the repository through one of the Organization's GitHub App connections, so the
command reads nothing of it on this machine. Cloud's preview of the manifest names the Agent and
the commit; the command creates that Agent where it doesn't exist, has Cloud build exactly that
commit, then follows the build and the rollout as a local deploy does.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from cayu.cli import _cloud_api, _cloud_project, main
from cayu.cli import cloud as cloud_cli
from cayu.cli._cloud_api import CloudApiClient

# The repository's name differs from the Agent its manifest names.
_REPOSITORY = "acme/desk-repo"
_AGENT = "support-desk"
_APP = f"/v1/applications/{_AGENT}"
_BUILD = f"{_APP}/deployments/repository"
_PREVIEW = ("POST", "/v1/repositories/manifest")
_LIST = ("GET", "/v1/applications")
_RELEASES = ("GET", f"{_APP}/deployments")
_SHA = "1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b"
_VERSION = f"1.0.0+{_SHA[:12]}"
_RELEASE_ID = "dep_repo1"
_DIGEST = "sha256:" + "c" * 64


class _Cloud:
    """Cloud's answers to each request, in order; the last one stands once the others are used."""

    def __init__(self, routes: dict[tuple[str, str], list[httpx.Response]]) -> None:
        self.routes = routes
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        answers = self.routes[(request.method, request.url.path)]
        return answers.pop(0) if len(answers) > 1 else answers[0]

    def sent(self, method: str, path: str) -> list[httpx.Request]:
        return [
            request
            for request in self.requests
            if (request.method, request.url.path) == (method, path)
        ]


@pytest.fixture
def cloud(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Callable[[dict[tuple[str, str], list[httpx.Response]]], _Cloud]:
    monkeypatch.setenv("CAYU_CLOUD_CONFIG", str(tmp_path / "cloud-config.json"))
    monkeypatch.setenv("CAYU_CLOUD_EVIDENCE_DIR", str(tmp_path / "evidence"))
    monkeypatch.delenv("CAYU_CLOUD_CONTEXT", raising=False)
    # A repository that isn't a local directory, whichever directory the tests run in.
    monkeypatch.chdir(tmp_path)

    def install(routes: dict[tuple[str, str], list[httpx.Response]]) -> _Cloud:
        fake = _Cloud(routes)
        real_client = httpx.Client
        monkeypatch.setattr(
            _cloud_api.httpx,
            "Client",
            lambda **kwargs: real_client(transport=httpx.MockTransport(fake), **kwargs),
        )
        monkeypatch.setattr(
            cloud_cli,
            "_cloud_client",
            lambda *_args, **_kwargs: CloudApiClient(
                api_url="https://cloud.example", api_key="login-token"
            ),
        )
        return fake

    return install


def _preview(
    *, ref: str | None = "main", connection: str = "acme-builds", application: str = _AGENT
) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "application": application,
            "connection": connection,
            "manifest_path": "cayu-cloud.toml",
            "name": "Support Desk",
            "ref": ref,
            "repository": _REPOSITORY,
            "revision": _SHA,
            "version": _VERSION,
        },
    )


def _release(
    *,
    status: str = "accepted",
    runtime: dict[str, object] | None = None,
    ref: str | None = None,
    connection: str = "acme-builds",
    **extra: object,
) -> dict[str, object]:
    return {
        "id": _RELEASE_ID,
        "status": status,
        "version": _VERSION,
        "manifest": {"runtime": runtime},
        "repository_source": {
            "archive_digest": _DIGEST,
            "commit_sha": _SHA,
            "connection": connection,
            "connection_id": "con_1",
            "manifest_path": "cayu-cloud.toml",
            "ref": ref,
            "repository": _REPOSITORY,
        },
        "resolved_revision": _SHA,
        **extra,
    }


def _agents(*ids: str) -> httpx.Response:
    return httpx.Response(
        200, json={"items": [{"id": item, "name": "Support Desk"} for item in ids]}
    )


def _refusal(status: int, code: str, message: str, **extra: object) -> httpx.Response:
    return httpx.Response(status, json={"detail": {"code": code, "message": message, **extra}})


_BARE_404 = httpx.Response(404, json={"detail": "Not Found"})


def _deploy(*options: str, source: str = _REPOSITORY) -> int:
    return main(["cloud", "deploy", source, *options])


def _output(capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
    return json.loads(capsys.readouterr().out)


def _sent(request: httpx.Request) -> dict[str, Any]:
    return json.loads(request.content)


def test_a_new_agent_is_the_one_the_manifest_names_and_builds_the_previewed_commit(
    cloud: Callable[..., _Cloud], capsys: pytest.CaptureFixture[str]
) -> None:
    fake = cloud(
        {
            _PREVIEW: [_preview()],
            _LIST: [_agents()],
            ("PUT", _APP): [httpx.Response(200, json={"id": _AGENT, "name": "Support Desk"})],
            ("POST", _BUILD): [httpx.Response(201, json=_release())],
        }
    )

    assert _deploy("--no-wait", source=f"https://github.com/{_REPOSITORY}") == 0

    captured = capsys.readouterr()
    output = json.loads(captured.out)
    assert [(request.method, request.url.path) for request in fake.requests] == [
        _PREVIEW,
        _LIST,
        ("PUT", _APP),
        ("POST", _BUILD),
    ]
    # No ref and no Agent: Cloud reads the default branch through a connection every Agent has.
    assert _sent(fake.requests[0]) == {
        "repository": _REPOSITORY,
        "manifest_path": "cayu-cloud.toml",
    }
    # Created as `deploy .` creates it, with the manifest's name.
    assert _sent(fake.sent("PUT", _APP)[0]) == {"name": "Support Desk"}
    (build,) = fake.sent("POST", _BUILD)
    # Exactly the commit and connection the preview read.
    assert _sent(build) == {
        "repository": _REPOSITORY,
        "manifest_path": "cayu-cloud.toml",
        "connection": "acme-builds",
        "revision": _SHA,
    }
    assert build.headers["Idempotency-Key"].startswith("deploy-repository:")
    assert f"read cayu-cloud.toml in {_REPOSITORY} at {_SHA} (main)" in captured.err
    assert f"it deploys {_AGENT}" in captured.err
    assert output["ok"] is True
    assert output["operation"] == "deploy"
    assert output["result"]["deployment"]["id"] == _RELEASE_ID
    assert output["result"]["source"] == {
        "connection": "acme-builds",
        "content_digest": _DIGEST,
        "kind": "repository_build",
        "manifest": "cayu-cloud.toml",
        # No ref was asked for: the default branch's commit is deployed by itself.
        "ref": None,
        "repository": f"https://github.com/{_REPOSITORY}",
        "revision": _SHA,
    }


def test_an_existing_agent_named_with_application_is_read_for_and_its_repeat_replays(
    cloud: Callable[..., _Cloud], capsys: pytest.CaptureFixture[str]
) -> None:
    # The Release as Cloud records it, with the acknowledgement and the session choice.
    release = _release(
        acknowledge_breaking=[7],
        session_policy={
            "acknowledge_sessions": [],
            "mode": "block",
            "source": "explicit",
            "wait_seconds": 900,
        },
    )
    fake = cloud(
        {
            _PREVIEW: [_preview(ref=None)],
            _LIST: [_agents(_AGENT)],
            ("POST", _BUILD): [httpx.Response(201, json=release)],
            ("GET", f"{_APP}/deployments/{_RELEASE_ID}"): [httpx.Response(200, json=release)],
        }
    )
    options = (
        "--application",
        _AGENT,
        "--revision",
        _SHA,
        "--connection",
        "acme-builds",
        "--manifest-path",
        "agents/desk/cayu-cloud.toml",
        "--acknowledge-breaking",
        "7",
        "--session-policy",
        "block",
        "--no-wait",
    )

    assert _deploy(*options) == 0
    capsys.readouterr()
    assert _deploy(*options) == 0

    assert fake.sent("PUT", _APP) == []
    preview = fake.sent(*_PREVIEW)[0]
    assert _sent(preview) == {
        "repository": _REPOSITORY,
        "manifest_path": "agents/desk/cayu-cloud.toml",
        "connection": "acme-builds",
        "revision": _SHA,
        "application": _AGENT,
    }
    first, second = fake.sent("POST", _BUILD)
    assert _sent(first) == {
        "repository": _REPOSITORY,
        "manifest_path": "agents/desk/cayu-cloud.toml",
        "connection": "acme-builds",
        "revision": _SHA,
        "acknowledge_breaking": [7],
        "session_policy": "block",
    }
    assert first.headers["Idempotency-Key"] == second.headers["Idempotency-Key"]
    assert _output(capsys)["result"]["source"]["ref"] is None


def test_application_for_an_agent_not_yet_created_must_match_the_manifest(
    cloud: Callable[..., _Cloud], capsys: pytest.CaptureFixture[str]
) -> None:
    fake = cloud({_PREVIEW: [_preview()], _LIST: [_agents()]})

    assert _deploy("--application", "other-agent", "--no-wait") == 2

    error = _output(capsys)["error"]
    assert error["code"] == "application_mismatch"
    assert error["application"] == _AGENT
    assert f"deploys {_AGENT}, not other-agent" in error["message"]
    assert f"--application {_AGENT}" in error["message"]
    # Read without the Agent, which doesn't exist, and nothing created.
    assert "application" not in _sent(fake.sent(*_PREVIEW)[0])
    assert [request.method for request in fake.requests] == ["GET", "POST"]


def test_clouds_application_mismatch_names_the_manifests_agent(
    cloud: Callable[..., _Cloud], capsys: pytest.CaptureFixture[str]
) -> None:
    cloud(
        {
            _PREVIEW: [_preview()],
            _LIST: [_agents(_AGENT)],
            ("POST", _BUILD): [
                _refusal(
                    409,
                    "application_mismatch",
                    f"cayu-cloud.toml deploys other-agent, not {_AGENT}.",
                    application="other-agent",
                )
            ],
        }
    )

    assert _deploy("--no-wait") == 2

    error = _output(capsys)["error"]
    assert error["code"] == "application_mismatch"
    assert error["application"] == "other-agent"
    assert "--application other-agent" in error["message"]


def test_cloud_chooses_the_connection_and_the_build_uses_the_one_it_chose(
    cloud: Callable[..., _Cloud], capsys: pytest.CaptureFixture[str]
) -> None:
    fake = cloud(
        {
            _PREVIEW: [_preview(connection="team-app")],
            _LIST: [_agents(_AGENT)],
            ("POST", _BUILD): [
                httpx.Response(201, json=_release(ref="release", connection="team-app"))
            ],
        }
    )

    assert _deploy("--ref", "release", "--no-wait") == 0

    captured = capsys.readouterr()
    preview = _sent(fake.sent(*_PREVIEW)[0])
    assert "connection" not in preview
    assert preview["ref"] == "release"
    build = _sent(fake.sent("POST", _BUILD)[0])
    assert build["connection"] == "team-app"
    # The previewed commit is built, and the ref asked for is recorded as its label.
    assert build["revision"] == _SHA
    assert build["ref"] == "release"
    assert "through the team-app connection" in captured.err
    source = json.loads(captured.out)["result"]["source"]
    assert source["connection"] == "team-app"
    assert source["ref"] == "release"


def test_the_build_and_the_rollout_are_followed_as_for_a_local_deploy(
    cloud: Callable[..., _Cloud], capsys: pytest.CaptureFixture[str]
) -> None:
    runtime: dict[str, object] = {
        "environment": {},
        "schedules": [],
        "web": {"command": "serve"},
        "worker": None,
    }
    service = {
        "application_id": _AGENT,
        "deployment_id": _RELEASE_ID,
        "status": "running",
        "application_url": "https://desk.example",
    }
    fake = cloud(
        {
            _PREVIEW: [_preview()],
            _LIST: [
                httpx.Response(
                    200,
                    json={
                        "items": [
                            {"id": _AGENT, "name": "Support Desk", "current_deployment_id": None}
                        ]
                    },
                )
            ],
            ("POST", _BUILD): [httpx.Response(201, json=_release(runtime=runtime))],
            ("GET", f"{_APP}/deployments/{_RELEASE_ID}"): [
                httpx.Response(200, json=_release(status="image_built", runtime=runtime)),
                httpx.Response(200, json=_release(status="promoted", runtime=runtime)),
            ],
            ("GET", f"{_APP}/service"): [httpx.Response(200, json=service)],
        }
    )

    assert _deploy("--poll-seconds", "0.01") == 0

    output = _output(capsys)
    assert output["result"]["deployment"]["status"] == "promoted"
    assert output["result"]["service"]["deployment_id"] == _RELEASE_ID
    assert output["result"]["service_publication_pending"] is False
    assert len(fake.sent("GET", f"{_APP}/deployments/{_RELEASE_ID}")) >= 2


def test_a_release_that_exists_is_found_by_its_version_and_followed(
    cloud: Callable[..., _Cloud], capsys: pytest.CaptureFixture[str]
) -> None:
    other = {"id": "dep_older", "status": "promoted", "version": "0.9.0+" + "d" * 12}
    fake = cloud(
        {
            _PREVIEW: [_preview()],
            _LIST: [_agents(_AGENT)],
            ("POST", _BUILD): [
                _refusal(409, "release_exists", "Already deployed.", version=_VERSION)
            ],
            _RELEASES: [
                httpx.Response(
                    200,
                    json={"items": [other, _release(status="image_built")], "next_cursor": None},
                )
            ],
        }
    )

    assert _deploy("--no-wait") == 0

    captured = capsys.readouterr()
    output = json.loads(captured.out)
    assert output["result"]["deployment"]["id"] == _RELEASE_ID
    # Said as a local deploy says it of unchanged source.
    assert output["result"]["unchanged_source"]["deployment_id"] == _RELEASE_ID
    assert len(fake.sent(*_RELEASES)) == 1


def test_a_release_followed_again_reports_its_own_source_not_the_previews(
    cloud: Callable[..., _Cloud], capsys: pytest.CaptureFixture[str]
) -> None:
    # Built earlier through another connection, under the ref's label.
    earlier = _release(status="promoted", connection="gh-all", ref="main")
    cloud(
        {
            _PREVIEW: [_preview(connection="gh-two")],
            _LIST: [_agents(_AGENT)],
            ("POST", _BUILD): [
                _refusal(409, "release_exists", "Already deployed.", version=_VERSION)
            ],
            _RELEASES: [httpx.Response(200, json={"items": [earlier], "next_cursor": None})],
        }
    )

    assert _deploy("--connection", "gh-two", "--ref", "main", "--no-wait") == 0

    source = _output(capsys)["result"]["source"]
    assert source["connection"] == "gh-all"
    assert source["ref"] == "main"
    assert source["revision"] == _SHA
    assert source["content_digest"] == _DIGEST


def test_a_release_that_exists_takes_an_acknowledgement_given_now(
    cloud: Callable[..., _Cloud], capsys: pytest.CaptureFixture[str]
) -> None:
    refused = {
        **_release(status="smoke_tested"),
        "publication_error": {
            "code": "breaking_revision_unacknowledged",
            "message": "Acknowledge storage revision 7.",
            "hint": "Deploy again with --acknowledge-breaking 7.",
        },
    }
    retried = {**_release(), "id": "dep_repo2"}
    fake = cloud(
        {
            _PREVIEW: [_preview()],
            _LIST: [_agents(_AGENT)],
            ("POST", _BUILD): [
                _refusal(409, "release_exists", "Already deployed.", version=_VERSION)
            ],
            _RELEASES: [httpx.Response(200, json={"items": [refused], "next_cursor": None})],
            ("GET", f"{_APP}/deployments/{_RELEASE_ID}"): [httpx.Response(200, json=refused)],
            ("POST", f"{_APP}/deployments/{_RELEASE_ID}/retry"): [
                httpx.Response(201, json=retried)
            ],
        }
    )

    assert _deploy("--acknowledge-breaking", "7", "--no-wait") == 0

    output = _output(capsys)
    (retry,) = fake.sent("POST", f"{_APP}/deployments/{_RELEASE_ID}/retry")
    assert _sent(retry) == {"acknowledge_breaking": [7]}
    assert output["result"]["retry"]["previous_deployment_id"] == _RELEASE_ID
    assert output["result"]["deployment"]["id"] == "dep_repo2"


@pytest.mark.parametrize(
    ("answer", "code", "said"),
    [
        (
            _refusal(
                422,
                "connection_required",
                f"No ready github-app connection is available to {_AGENT}.",
                connections=[],
            ),
            "connection_required",
            "cayu cloud connections add github-app --name NAME",
        ),
        (
            _refusal(
                422,
                "connection_ambiguous",
                "Several connections can each read the repository.",
                connections=["acme-builds", "team-app"],
            ),
            "connection_ambiguous",
            "Pass --connection with one of: acme-builds, team-app.",
        ),
        (
            _refusal(422, "repository_not_installed", f"The App isn't installed on {_REPOSITORY}."),
            "repository_not_installed",
            f"Install the connection's GitHub App on {_REPOSITORY}",
        ),
        (
            _refusal(422, "source_not_found", f"{_REPOSITORY} has no ref nightly."),
            "source_not_found",
            "Check the repository, --ref or --revision, and --manifest-path.",
        ),
        (
            _refusal(413, "source_too_large", "The source is larger than Cloud builds from."),
            "source_too_large",
            "Deploy a smaller subdirectory with --manifest-path",
        ),
        (
            _refusal(504, "source_fetch_timeout", "GitHub didn't send the source in time."),
            "source_fetch_timeout",
            "Run the same command again.",
        ),
        (
            _refusal(429, "too_many_calls", "GitHub limits calls right now."),
            "too_many_calls",
            "Run the same command again in a minute.",
        ),
        (
            _refusal(422, "scope_not_allowed", "The connection can't read contents."),
            "scope_not_allowed",
            "Contents: read permission",
        ),
        (
            _refusal(404, "connection_not_found", "There is no connection named nightly."),
            "connection_not_found",
            "`cayu cloud connections list`",
        ),
    ],
)
def test_each_refusal_says_what_to_do_next_and_creates_nothing(
    cloud: Callable[..., _Cloud],
    capsys: pytest.CaptureFixture[str],
    answer: httpx.Response,
    code: str,
    said: str,
) -> None:
    fake = cloud({_PREVIEW: [answer]})

    assert _deploy("--no-wait") == 2

    error = _output(capsys)["error"]
    assert error["category"] == "repository_deploy_refused"
    assert error["code"] == code
    assert said in error["message"]
    # A keyed 404 is Cloud's refusal, never an older Cloud.
    assert [request.url.path for request in fake.requests] == [_PREVIEW[1]]


def test_connections_only_for_named_agents_ask_for_an_application(
    cloud: Callable[..., _Cloud], capsys: pytest.CaptureFixture[str]
) -> None:
    cloud(
        {
            _PREVIEW: [
                _refusal(
                    422,
                    "connection_required",
                    "No ready github-app connection is available to every Agent.",
                    connections=["desk-builds"],
                )
            ]
        }
    )

    assert _deploy("--no-wait") == 2

    error = _output(capsys)["error"]
    assert error["connections"] == ["desk-builds"]
    assert (
        "These GitHub App connections exist but can't be used here (not ready, or available "
        "only to some Agents): desk-builds; pass --application AGENT or make one available to "
        "every Agent."
    ) in error["message"]
    assert "connections add" not in error["message"]


def test_a_bare_404_from_the_deploy_route_is_an_error_not_an_older_cloud(
    cloud: Callable[..., _Cloud], capsys: pytest.CaptureFixture[str]
) -> None:
    cloud(
        {
            _PREVIEW: [_preview()],
            _LIST: [_agents(_AGENT)],
            ("POST", _BUILD): [_BARE_404],
        }
    )

    assert _deploy("--revision", _SHA, "--no-wait") == 2

    error = _output(capsys)["error"]
    assert error["category"] == "api_request_rejected"
    assert "404" in error["message"]


@pytest.mark.parametrize(
    "options",
    [("--ref", "main"), ("--revision", _SHA, "--connection", "acme-builds"), ()],
    ids=["ref", "connection", "default-branch"],
)
def test_an_older_cloud_refuses_a_ref_or_a_connection_before_anything_is_created(
    cloud: Callable[..., _Cloud], capsys: pytest.CaptureFixture[str], options: tuple[str, ...]
) -> None:
    fake = cloud({_PREVIEW: [_BARE_404]})

    assert _deploy(*options, "--no-wait") == 2

    error = _output(capsys)["error"]
    assert error["category"] == "repository_build_unsupported"
    assert "can't build from a repository" in error["message"]
    assert [request.url.path for request in fake.requests] == [_PREVIEW[1]]


def test_an_older_cloud_deploys_an_exact_commit_as_before(
    cloud: Callable[..., _Cloud],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = "\n".join(
        [
            "schema_version = 1",
            f'application = "{_AGENT}"',
            'name = "Support Desk"',
            'version = "1.0.0"',
            'entrypoint = "python app.py"',
            'capabilities = ["network"]',
            "cpu_millis = 1000",
            "memory_mb = 2048",
            "timeout_seconds = 600",
            'environment = "python"',
            'compatibility = "cayu>=0.1"',
            'policy_version = "v1"',
        ]
    )
    read: list[tuple[str, str, str]] = []

    def github_file(repository: str, *, revision: str, path: str) -> str:
        read.append((repository, revision, path))
        return manifest

    monkeypatch.setattr(_cloud_project, "_github_file", github_file)
    fake = cloud(
        {
            _PREVIEW: [_BARE_404],
            _LIST: [_agents()],
            ("PUT", _APP): [httpx.Response(200, json={"id": _AGENT, "name": "Support Desk"})],
            ("POST", f"{_APP}/deployments"): [
                httpx.Response(
                    201,
                    json={
                        "id": "dep_old",
                        "status": "accepted",
                        "candidate": {"evaluated_against_deployment_id": None},
                    },
                )
            ],
        }
    )

    # `--no-promote` stays as it was on such a Cloud.
    assert (
        _deploy("--revision", _SHA, "--no-wait", "--no-promote", "--evaluated-against", "none") == 0
    )

    captured = capsys.readouterr()
    assert read == [(f"https://github.com/{_REPOSITORY}", _SHA, "cayu-cloud.toml")]
    assert "with your GitHub login" in captured.err
    # The Agent is the manifest's, as before.
    assert _sent(fake.sent("PUT", _APP)[0]) == {"name": "Support Desk"}
    (created,) = fake.sent("POST", f"{_APP}/deployments")
    assert _sent(created)["manifest"]["source"] == {
        "repository": f"https://github.com/{_REPOSITORY}",
        "revision": _SHA,
    }
    assert _sent(created)["candidate"] == {"evaluated_against_deployment_id": None}
    assert json.loads(captured.out)["result"]["source"]["kind"] == "repository"


def test_no_promote_is_refused_by_a_cloud_that_builds_before_anything_is_created(
    cloud: Callable[..., _Cloud], capsys: pytest.CaptureFixture[str]
) -> None:
    fake = cloud({_PREVIEW: [_preview()]})

    assert _deploy("--no-promote", "--evaluated-against", "none", "--no-wait") == 2

    assert _output(capsys)["error"]["category"] == "invalid_input"
    assert [request.url.path for request in fake.requests] == [_PREVIEW[1]]


def test_a_manifest_not_named_cayu_cloud_toml_is_explained(
    cloud: Callable[..., _Cloud], capsys: pytest.CaptureFixture[str]
) -> None:
    rejected = httpx.Response(
        422, json={"detail": [{"loc": ["body"], "type": "value_error", "msg": "bad"}]}
    )
    cloud({_PREVIEW: [rejected]})

    assert _deploy("--manifest-path", "deploy.toml", "--no-wait") == 2

    error = _output(capsys)["error"]
    assert error["category"] == "manifest_invalid"
    assert "cayu-cloud.toml" in error["message"]


@pytest.mark.parametrize(
    ("options", "category"),
    [
        (("--ref", "main", "--revision", _SHA), "source_invalid"),
        (("--revision", "main"), "source_revision_invalid"),
        (("--manifest-path", "../cayu-cloud.toml"), "manifest_invalid"),
    ],
)
def test_a_bad_repository_deploy_is_refused_before_anything_is_sent(
    cloud: Callable[..., _Cloud],
    capsys: pytest.CaptureFixture[str],
    options: tuple[str, ...],
    category: str,
) -> None:
    fake = cloud({})

    assert _deploy(*options, "--no-wait") == 2

    assert _output(capsys)["error"]["category"] == category
    assert fake.requests == []


def test_a_ref_or_connection_is_refused_for_a_local_directory(
    cloud: Callable[..., _Cloud], capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    fake = cloud({})

    assert _deploy("--ref", "main", source=str(tmp_path)) == 2

    assert _output(capsys)["error"]["category"] == "source_invalid"
    assert fake.requests == []


def test_too_many_previews_says_when_to_run_again(
    cloud: Callable[..., _Cloud], capsys: pytest.CaptureFixture[str]
) -> None:
    limited = httpx.Response(
        429,
        headers={"Retry-After": "17"},
        json={
            "detail": {
                "code": "too_many_previews",
                "message": "This Organization previews at most 30 manifests a minute.",
            }
        },
    )
    fake = cloud({_PREVIEW: [limited]})

    assert _deploy("--no-wait") == 2

    error = _output(capsys)["error"]
    assert error["code"] == "too_many_previews"
    assert error["message"] == (
        "This Organization previews at most 30 manifests a minute. "
        "Run the same command again in 17 seconds."
    )
    assert [request.url.path for request in fake.requests] == [_PREVIEW[1]]


def test_a_404_that_isnt_the_frameworks_own_is_no_older_cloud(
    cloud: Callable[..., _Cloud], capsys: pytest.CaptureFixture[str]
) -> None:
    cloud({_PREVIEW: [httpx.Response(404)]})

    assert _deploy("--revision", _SHA, "--no-wait") == 2

    error = _output(capsys)["error"]
    assert error["category"] == "api_request_rejected"
    assert "404" in error["message"]


def test_a_plain_validation_refusal_shows_clouds_reason(
    cloud: Callable[..., _Cloud], capsys: pytest.CaptureFixture[str]
) -> None:
    cloud({_PREVIEW: [httpx.Response(422, json={"detail": "Give ref or revision, not both."})]})

    assert _deploy("--no-wait") == 2

    error = _output(capsys)["error"]
    assert error["category"] == "repository_request_invalid"
    assert error["message"] == "Cayu Cloud refused the request: Give ref or revision, not both."


def test_validation_errors_from_the_deploy_route_are_shown_bounded(
    cloud: Callable[..., _Cloud], capsys: pytest.CaptureFixture[str]
) -> None:
    errors = [
        {"loc": ["body"], "type": "value_error", "msg": "Value error, Give ref or revision."},
        *({"loc": ["body", "ref"], "type": "string_too_long", "msg": "x " * 120} for _ in range(4)),
    ]
    cloud(
        {
            _PREVIEW: [_preview()],
            _LIST: [_agents(_AGENT)],
            ("POST", _BUILD): [httpx.Response(422, json={"detail": errors})],
        }
    )

    assert _deploy("--no-wait") == 2

    message = _output(capsys)["error"]["message"]
    assert message.startswith(
        "Cayu Cloud refused the request: Value error, Give ref or revision.; x x"
    )
    assert message.endswith("...")
    assert len(message) <= len("Cayu Cloud refused the request: ") + 512


def test_a_ref_and_the_same_commit_by_revision_are_different_requests(
    cloud: Callable[..., _Cloud], capsys: pytest.CaptureFixture[str]
) -> None:
    fake = cloud(
        {
            _PREVIEW: [_preview(), _preview(ref=None)],
            _LIST: [_agents(_AGENT)],
            ("POST", _BUILD): [httpx.Response(201, json=_release())],
        }
    )

    assert _deploy("--ref", "main", "--no-wait") == 0
    capsys.readouterr()
    assert _deploy("--revision", _SHA, "--no-wait") == 0
    capsys.readouterr()

    by_ref, by_revision = fake.sent("POST", _BUILD)
    assert _sent(by_ref)["ref"] == "main"
    assert "ref" not in _sent(by_revision)
    assert by_ref.headers["Idempotency-Key"] != by_revision.headers["Idempotency-Key"]


@pytest.mark.parametrize(
    "conflict",
    [
        httpx.Response(
            409,
            json={"detail": "Idempotency-Key was already used for a different deployment request."},
        ),
        _refusal(409, "idempotency_conflict", "The key was already used."),
    ],
    ids=["plain", "coded"],
)
def test_a_key_used_before_follows_the_release_of_that_commit(
    cloud: Callable[..., _Cloud], capsys: pytest.CaptureFixture[str], conflict: httpx.Response
) -> None:
    fake = cloud(
        {
            _PREVIEW: [_preview()],
            _LIST: [_agents(_AGENT)],
            ("POST", _BUILD): [conflict],
            _RELEASES: [
                httpx.Response(
                    200, json={"items": [_release(status="promoted")], "next_cursor": None}
                )
            ],
        }
    )

    assert _deploy("--no-wait") == 0

    output = _output(capsys)
    assert output["result"]["deployment"]["id"] == _RELEASE_ID
    assert output["result"]["unchanged_source"]["deployment_id"] == _RELEASE_ID
    assert len(fake.sent(*_RELEASES)) == 1


def test_a_release_an_earlier_cli_deployed_is_followed_with_its_recorded_source(
    cloud: Callable[..., _Cloud], capsys: pytest.CaptureFixture[str]
) -> None:
    # Deployed by an earlier CLI's --revision: the same version, no repository_source.
    earlier = {
        "id": _RELEASE_ID,
        "status": "promoted",
        "version": _VERSION,
        "manifest": {
            "runtime": None,
            "source": {"repository": f"https://github.com/{_REPOSITORY}", "revision": _SHA},
        },
    }
    cloud(
        {
            _PREVIEW: [_preview(ref=None)],
            _LIST: [_agents(_AGENT)],
            ("POST", _BUILD): [
                _refusal(409, "release_exists", "Already deployed.", version=_VERSION)
            ],
            _RELEASES: [httpx.Response(200, json={"items": [earlier], "next_cursor": None})],
        }
    )

    assert _deploy("--revision", _SHA, "--no-wait") == 0

    assert _output(capsys)["result"]["source"] == {
        "connection": None,
        "content_digest": None,
        "kind": "repository",
        "manifest": None,
        "ref": None,
        "repository": f"https://github.com/{_REPOSITORY}",
        "revision": _SHA,
    }
