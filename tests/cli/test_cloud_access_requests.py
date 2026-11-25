"""`cayu cloud access-requests submit` asks for access to Cayu Cloud without a credential."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from tests.cli.test_cloud_api_keys import cloud as _cloud

from cayu.cli import main

cloud = _cloud

_SUBMIT = "/v1/access-requests"


def _run(arguments: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, dict[str, Any]]:
    code = main(["cloud", "access-requests", *arguments])
    return code, json.loads(capsys.readouterr().out)


def _submit(*extra: str) -> list[str]:
    return [
        "submit",
        "--name",
        "Ada Lovelace",
        "--email",
        "ada@example.com",
        "--company",
        "Example Co",
        *extra,
    ]


def test_submit_sends_the_request_without_a_credential(
    cloud, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = cloud({("POST", _SUBMIT): httpx.Response(202, json={"status": "received"})})
    monkeypatch.setenv("CAYU_CLOUD_API_URL", "https://cloud.example")

    code, output = _run(_submit("--message", "We run support Agents."), capsys)

    assert code == 0
    assert output == {
        "ok": True,
        "operation": "access-requests.submit",
        "result": {"status": "received"},
    }
    (request,) = fake.requests
    assert str(request.url) == "https://cloud.example/v1/access-requests"
    assert "authorization" not in request.headers
    assert json.loads(request.content) == {
        "email": "ada@example.com",
        "message": "We run support Agents.",
        "name": "Ada Lovelace",
        "organization_name": "Example Co",
    }


def test_submit_needs_no_login_and_goes_to_production_by_default(
    cloud, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from cayu.cli import cloud as cloud_cli

    fake = cloud({("POST", _SUBMIT): httpx.Response(202, json={"status": "received"})})
    monkeypatch.delenv("CAYU_CLOUD_API_URL", raising=False)

    def no_login(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("submit must not sign in")

    monkeypatch.setattr(cloud_cli, "_cloud_client", no_login)

    code, output = _run(_submit(), capsys)

    assert code == 0, output
    (request,) = fake.requests
    assert str(request.url) == "https://cloud.cayu.dev/v1/access-requests"
    assert "message" not in json.loads(request.content)


@pytest.mark.parametrize(
    ("flag", "value"),
    [("--email", "ada.example.com"), ("--email", "@example.com"), ("--name", "  ")],
)
def test_submit_refuses_what_cloud_would_before_sending(
    cloud, capsys: pytest.CaptureFixture[str], flag: str, value: str
) -> None:
    fake = cloud({})
    arguments = _submit()
    arguments[arguments.index(flag) + 1] = value

    code, output = _run(arguments, capsys)

    assert code == 2
    assert output["error"]["category"] == "invalid_input"
    assert flag in output["error"]["message"]
    assert fake.requests == []


def test_submit_reports_cloud_refusing_it(cloud, capsys: pytest.CaptureFixture[str]) -> None:
    cloud({("POST", _SUBMIT): httpx.Response(422, json={"detail": "Enter a valid work email."})})

    code, output = _run(_submit(), capsys)

    assert code == 2
    assert output["error"]["category"] == "api_request_rejected"
    assert "HTTP 422" in output["error"]["message"]


def test_submit_acts_in_no_organization(
    cloud, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = cloud({("POST", _SUBMIT): httpx.Response(202, json={"status": "received"})})

    code = main(["cloud", "--organization", "org_other", "access-requests", *_submit()])
    output = json.loads(capsys.readouterr().out)

    assert code == 2
    assert output["error"]["category"] == "invalid_input"
    assert "access-requests" in output["error"]["message"]
    assert fake.requests == []

    # A shell's Organization is ignored, even a malformed one.
    monkeypatch.setenv("CAYU_CLOUD_ORGANIZATION", "not an id")
    code, output = _run(_submit(), capsys)
    assert code == 0, output
    assert "organization" not in output
    assert len(fake.requests) == 1


def test_submit_has_no_organization_flag(capsys: pytest.CaptureFixture[str]) -> None:
    code, output = _run([*_submit(), "--organization", "org_1"], capsys)

    assert code == 2
    assert output["error"]["category"] == "invalid_input"


def test_only_submit_is_offered(capsys: pytest.CaptureFixture[str]) -> None:
    code, output = _run(["list"], capsys)

    assert code == 2
    assert output["error"]["category"] == "invalid_input"
