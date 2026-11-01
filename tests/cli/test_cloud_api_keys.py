"""`cayu cloud api-keys create|list|revoke` over Cloud's Organization API key routes."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from cayu.cli import _cloud_api, main
from cayu.cli import cloud as cloud_cli
from cayu.cli._cloud_api import CloudApiClient

_KEYS = "/v1/organization/api-keys"


def _key(**changes: object) -> dict[str, Any]:
    return {
        "id": "key_1",
        "tenant_id": "org_1",
        "name": "github-actions",
        "scopes": ["deploy", "read"],
        "scoped": True,
        "active": True,
        "created_at": "2026-10-31T12:00:00+00:00",
        **changes,
    }


_Answer = httpx.Response | Callable[[httpx.Request], httpx.Response]


class _Cloud:
    def __init__(self, routes: dict[tuple[str, str], _Answer]) -> None:
        self.routes = routes
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        answer = self.routes[(request.method, request.url.raw_path.decode())]
        return answer if isinstance(answer, httpx.Response) else answer(request)


@pytest.fixture
def cloud(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Callable[[dict[tuple[str, str], _Answer]], _Cloud]:
    monkeypatch.setenv("CAYU_CLOUD_CONFIG", str(tmp_path / "cloud-config.json"))
    monkeypatch.delenv("CAYU_CLOUD_CONTEXT", raising=False)

    def install(routes: dict[tuple[str, str], _Answer]) -> _Cloud:
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


def _run(arguments: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, dict[str, Any]]:
    code = main(["cloud", "api-keys", *arguments])
    return code, json.loads(capsys.readouterr().out)


def test_create_writes_a_private_key_file_and_sends_only_its_digest(
    cloud, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    fake = cloud({("POST", _KEYS): httpx.Response(201, json=_key())})
    key_file = tmp_path / "ci-key"

    code, output = _run(
        ["create", "--name", "github-actions", "--purpose", "ci", "--key-file", str(key_file)],
        capsys,
    )

    assert code == 0
    token = key_file.read_text().strip()
    assert token.startswith("cayu_")
    assert key_file.stat().st_mode & 0o777 == 0o600
    (request,) = fake.requests
    assert json.loads(request.content) == {
        "name": "github-actions",
        "purpose": "ci",
        "api_key_digest": hashlib.sha256(token.encode()).hexdigest(),
    }
    assert request.headers["authorization"] == "Bearer login-token"
    assert output["operation"] == "api-keys.create"
    assert output["result"]["id"] == "key_1"
    assert output["result"]["key_file"] == str(key_file)
    printed = json.dumps(output)
    assert token not in printed
    assert hashlib.sha256(token.encode()).hexdigest() not in printed


def test_create_names_scopes_instead_of_a_purpose(
    cloud, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    fake = cloud({("POST", _KEYS): httpx.Response(201, json=_key(scopes=["read"]))})

    code, _ = _run(
        [
            "create",
            "--name",
            "reader",
            "--scope",
            "read",
            "--scope",
            "report",
            "--key-file",
            str(tmp_path / "reader-key"),
        ],
        capsys,
    )

    assert code == 0
    body = json.loads(fake.requests[0].content)
    assert body["scopes"] == ["read", "report"]
    assert "purpose" not in body


def test_create_refuses_an_existing_key_file_before_asking_cloud(
    cloud, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    fake = cloud({})
    key_file = tmp_path / "ci-key"
    key_file.write_text("keep me\n")

    code, output = _run(
        ["create", "--name", "ci", "--purpose", "ci", "--key-file", str(key_file)], capsys
    )

    assert code == 2
    assert output["error"]["category"] == "invalid_input"
    assert "already exists" in output["error"]["message"]
    assert key_file.read_text() == "keep me\n"
    assert fake.requests == []


def test_a_refused_create_removes_the_key_file(
    cloud, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    cloud(
        {
            ("POST", _KEYS): httpx.Response(
                403,
                json={
                    "detail": {
                        "code": "admin_required",
                        "message": "Only an organization admin can do this.",
                    }
                },
            )
        }
    )
    key_file = tmp_path / "ci-key"

    code, output = _run(
        ["create", "--name", "ci", "--purpose", "ci", "--key-file", str(key_file)], capsys
    )

    assert code == 2
    assert output["ok"] is False
    assert not key_file.exists()


def _lost_answer(request: httpx.Request) -> httpx.Response:
    raise httpx.ReadTimeout("no answer", request=request)


@pytest.mark.parametrize(
    "answer",
    [_lost_answer, httpx.Response(500, json={"detail": "boom"}), httpx.Response(201, text="{")],
    ids=["lost", "server-error", "unreadable"],
)
def test_a_create_whose_answer_may_follow_a_kept_key_keeps_the_key_file(
    cloud, capsys: pytest.CaptureFixture[str], tmp_path: Path, answer: _Answer
) -> None:
    stored: list[str] = []

    def route(request: httpx.Request) -> httpx.Response:
        # Cloud keeps the digest before its answer goes missing.
        stored.append(json.loads(request.content)["api_key_digest"])
        return answer if isinstance(answer, httpx.Response) else answer(request)

    cloud({("POST", _KEYS): route})
    key_file = tmp_path / "ci-key"

    code, output = _run(
        ["create", "--name", "ci", "--purpose", "ci", "--key-file", str(key_file)], capsys
    )

    assert code == 2
    token = key_file.read_text().strip()
    assert stored == [hashlib.sha256(token.encode()).hexdigest()]
    assert "--retry" in output["error"]["message"]
    assert token not in json.dumps(output)


def test_retry_sends_the_kept_key_again(
    cloud, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    fake = cloud({("POST", _KEYS): httpx.Response(201, json=_key())})
    key_file = tmp_path / "ci-key"
    key_file.write_text("cayu_kept-token\n")

    code, output = _run(
        ["create", "--name", "ci", "--purpose", "ci", "--key-file", str(key_file), "--retry"],
        capsys,
    )

    assert code == 0
    assert json.loads(fake.requests[0].content)["api_key_digest"] == (
        hashlib.sha256(b"cayu_kept-token").hexdigest()
    )
    assert output["result"]["id"] == "key_1"
    assert key_file.read_text() == "cayu_kept-token\n"


def test_a_refused_retry_keeps_the_key_file_and_a_bad_one_is_refused(
    cloud, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    fake = cloud({("POST", _KEYS): httpx.Response(409, json={"detail": {"code": "x"}})})
    key_file = tmp_path / "ci-key"
    key_file.write_text("cayu_kept-token\n")
    retry = ["create", "--name", "ci", "--purpose", "ci", "--key-file", str(key_file), "--retry"]

    code, _ = _run(retry, capsys)
    assert code == 2
    assert key_file.exists()

    key_file.write_text("not a key\n")
    code, output = _run(retry, capsys)
    assert code == 2
    assert output["error"]["category"] == "invalid_input"
    assert len(fake.requests) == 1


def test_create_needs_a_purpose_or_scopes_not_both(
    cloud, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    fake = cloud({})
    key_file = tmp_path / "k"
    for use in ([], ["--purpose", "ci", "--scope", "read"]):
        code, output = _run(["create", "--name", "ci", *use, "--key-file", str(key_file)], capsys)
        assert code == 2
        assert output["error"]["category"] == "invalid_input"
    assert fake.requests == []
    assert not key_file.exists()


def test_list_and_revoke(cloud, capsys: pytest.CaptureFixture[str]) -> None:
    fake = cloud(
        {
            ("GET", _KEYS): httpx.Response(200, json={"items": [_key()]}),
            ("DELETE", f"{_KEYS}/key_1"): httpx.Response(200, json=_key(active=False)),
        }
    )

    code, listed = _run(["list"], capsys)
    assert code == 0
    assert listed["operation"] == "api-keys.list"
    assert listed["result"]["items"][0]["name"] == "github-actions"

    code, revoked = _run(["revoke", "key_1"], capsys)
    assert code == 0
    assert revoked["operation"] == "api-keys.revoke"
    assert revoked["result"]["active"] is False
    assert [request.method for request in fake.requests] == ["GET", "DELETE"]
