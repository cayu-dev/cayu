"""`cayu cloud connectors` and `cayu cloud connections` over Cloud's connection routes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
from tests.cli.test_cloud_api_keys import cloud as _cloud

from cayu.cli import main

if TYPE_CHECKING:
    import pytest

cloud = _cloud

_APPS = "/v1/applications"
_CONNECTION = {"name": "team-anthropic", "connector": "anthropic", "status": "ready"}
_AGENTS = {"items": [{"id": "support-desk", "name": "Support desk"}]}


def _run(arguments: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, dict[str, Any]]:
    code = main(["cloud", *arguments])
    return code, json.loads(capsys.readouterr().out)


def test_add_reads_the_key_from_a_file_and_never_prints_it(
    cloud, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    fake = cloud(
        {
            ("GET", f"{_APPS}?lifecycle=all"): httpx.Response(200, json=_AGENTS),
            ("POST", "/v1/connections"): httpx.Response(201, json=_CONNECTION),
        }
    )
    key_file = tmp_path / "anthropic-key"
    key_file.write_text("sk-ant-secret\n")

    code, output = _run(
        [
            "connections",
            "add",
            "anthropic",
            "--name",
            "team-anthropic",
            "--key-file",
            str(key_file),
            "--agent",
            "Support desk",
        ],
        capsys,
    )

    assert code == 0
    (post,) = (request for request in fake.requests if request.method == "POST")
    assert json.loads(post.content) == {
        "connector": "anthropic",
        "name": "team-anthropic",
        "credential": {"api_key": "sk-ant-secret"},
        "agents": ["support-desk"],
    }
    assert output["operation"] == "connections.add"
    assert "sk-ant-secret" not in json.dumps(output)


def test_a_key_pair_takes_its_secret_field_from_a_file(
    cloud, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    fake = cloud({("POST", "/v1/connections"): httpx.Response(201, json=_CONNECTION)})
    private_key = tmp_path / "snowflake.p8"
    private_key.write_text("-----BEGIN PRIVATE KEY-----\n")

    code, _ = _run(
        [
            "connections",
            "add",
            "snowflake",
            "--name",
            "warehouse",
            "--credential",
            "account=acme",
            "--credential",
            "user=cayu",
            "--credential",
            f"private_key=@{private_key}",
        ],
        capsys,
    )

    assert code == 0
    assert json.loads(fake.requests[0].content)["credential"] == {
        "account": "acme",
        "user": "cayu",
        "private_key": "-----BEGIN PRIVATE KEY-----",
    }


def test_an_unreadable_credential_file_asks_cloud_nothing(
    cloud, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    fake = cloud({})

    code, output = _run(
        [
            "connections",
            "add",
            "anthropic",
            "--name",
            "team-anthropic",
            "--key-file",
            str(tmp_path / "missing"),
        ],
        capsys,
    )

    assert code == 2
    assert output["error"]["category"] == "invalid_input"
    assert "missing" not in output["error"]["message"]
    assert fake.requests == []


def test_a_secret_field_given_on_the_command_line_is_refused_unrepeated(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = cloud({})

    for given in ("private_key=sk-live-secret", "sk-live-secret", "sk_live_secret=x"):
        code, output = _run(
            ["connections", "add", "snowflake", "--name", "warehouse", "--credential", given],
            capsys,
        )
        assert code == 2
        assert "sk-live-secret" not in json.dumps(output)
    assert fake.requests == []


def test_many_agents_are_resolved_from_one_list(
    cloud, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    key_file = tmp_path / "anthropic-key"
    key_file.write_text("sk-ant-secret\n")
    agents = {
        "items": [
            {"id": "support-desk", "name": "Support desk"},
            {"id": "triage", "name": "Triage"},
        ]
    }
    fake = cloud(
        {
            ("GET", f"{_APPS}?lifecycle=all"): httpx.Response(200, json=agents),
            ("POST", "/v1/connections"): httpx.Response(201, json=_CONNECTION),
        }
    )

    code, _ = _run(
        [
            "connections",
            "add",
            "anthropic",
            "--name",
            "team-anthropic",
            "--key-file",
            str(key_file),
            "--agent",
            "Support desk",
            "--agent",
            "Triage",
        ],
        capsys,
    )

    assert code == 0
    lists = [request for request in fake.requests if request.url.path == _APPS]
    assert len(lists) == 1
    (post,) = (request for request in fake.requests if request.method == "POST")
    assert json.loads(post.content)["agents"] == ["support-desk", "triage"]


def test_agents_all_makes_it_available_to_every_agent(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = cloud(
        {("PUT", "/v1/connections/team-anthropic/agents"): httpx.Response(200, json=_CONNECTION)}
    )

    code, _ = _run(["connections", "agents", "team-anthropic", "--all"], capsys)

    assert code == 0
    assert json.loads(fake.requests[0].content) == {"agents": None}


def test_bind_and_unbind_an_agents_slot(cloud, capsys: pytest.CaptureFixture[str]) -> None:
    slot = f"{_APPS}/support-desk/connections/primary"
    fake = cloud(
        {
            ("GET", f"{_APPS}?lifecycle=all"): httpx.Response(200, json=_AGENTS),
            ("PUT", slot): httpx.Response(200, json={"slot": "primary"}),
            ("DELETE", slot): httpx.Response(204),
        }
    )

    code, _ = _run(
        [
            "connections",
            "bind",
            "primary",
            "team-anthropic",
            "--application",
            "Support desk",
            "--scope",
            "read",
        ],
        capsys,
    )
    assert code == 0
    (put,) = (request for request in fake.requests if request.method == "PUT")
    assert json.loads(put.content) == {
        "connection": "team-anthropic",
        "scopes": ["read"],
    }

    code, output = _run(
        ["connections", "unbind", "primary", "--application", "support-desk"], capsys
    )
    assert code == 0
    assert output["result"] == {
        "application_id": "support-desk",
        "slot": "primary",
        "unbound": True,
    }


def test_a_refused_removal_reports_clouds_reason(cloud, capsys: pytest.CaptureFixture[str]) -> None:
    cloud(
        {
            ("DELETE", "/v1/connections/team-anthropic"): httpx.Response(
                409,
                json={
                    "detail": {
                        "code": "connection_bound",
                        "message": "Agents support-desk are bound to this connection.",
                    }
                },
            )
        }
    )

    code, output = _run(["connections", "remove", "team-anthropic"], capsys)

    assert code != 0
    assert output["error"]["code"] == "connection_bound"
    assert "support-desk" in output["error"]["message"]
