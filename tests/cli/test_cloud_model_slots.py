"""`cayu cloud models list` and `connections bind|unbind --model` over Cloud's model routes."""

from __future__ import annotations

import json

import httpx
import pytest
from tests.cli.test_cloud_api_keys import cloud as _cloud

from cayu.cli import main

cloud = _cloud

_APPS = "/v1/applications"
_AGENTS = {"items": [{"id": "support-desk", "name": "Support desk"}]}
_MODELS = f"{_APPS}/support-desk/models"
_LISTING = {
    "release_id": "rel_1",
    "approved_models": ["bedrock/claude-opus-4-1", "bedrock/claude-sonnet-4-5"],
    "items": [
        {
            "slot": "primary",
            "model": "bedrock/claude-opus-4-1",
            "source": "bound",
            "default": "bedrock/claude-sonnet-4-5",
            "label": "Primary model",
            "declared": True,
            "approved": True,
            "bound_by": "user_1",
            "bound_at": "2026-11-10T12:00:00Z",
        },
        {
            "slot": "fast",
            "model": None,
            "source": "unbound",
            "default": None,
            "label": None,
            "declared": False,
            "approved": None,
        },
    ],
}


def test_models_list_prints_a_table_or_the_json_envelope(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = cloud(
        {
            ("GET", f"{_APPS}?lifecycle=all"): httpx.Response(200, json=_AGENTS),
            ("GET", _MODELS): httpx.Response(200, json=_LISTING),
        }
    )

    assert main(["cloud", "models", "list", "--application", "Support desk"]) == 0
    table = capsys.readouterr().out.splitlines()
    assert table[0].split() == ["SLOT", "MODEL", "SOURCE", "APPROVED", "DECLARED"]
    assert table[1].split() == ["primary", "bedrock/claude-opus-4-1", "bound", "yes", "yes"]
    assert table[2].split() == ["fast", "-", "unbound", "unknown", "no"]
    assert table[-1] == ("Approved models: bedrock/claude-opus-4-1, bedrock/claude-sonnet-4-5")

    assert main(["cloud", "models", "list", "--application", "support-desk", "--json"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["operation"] == "models.list"
    assert output["result"] == _LISTING
    assert [request.method for request in fake.requests].count("GET") == 4


def test_bind_and_unbind_a_model_slot(cloud, capsys: pytest.CaptureFixture[str]) -> None:
    slot = f"{_MODELS}/primary"
    fake = cloud(
        {
            ("GET", f"{_APPS}?lifecycle=all"): httpx.Response(200, json=_AGENTS),
            ("PUT", slot): httpx.Response(200, json=_LISTING["items"][0]),
            ("DELETE", slot): httpx.Response(204),
        }
    )

    arguments = ["connections", "bind", "--application", "support-desk", "primary"]
    assert main(["cloud", *arguments, "--model", "bedrock/claude-opus-4-1"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["operation"] == "models.bind"
    assert output["result"]["model"] == "bedrock/claude-opus-4-1"
    (put,) = (request for request in fake.requests if request.method == "PUT")
    assert json.loads(put.content) == {"model": "bedrock/claude-opus-4-1"}

    assert (
        main(
            [
                "cloud",
                "connections",
                "unbind",
                "primary",
                "--application",
                "support-desk",
                "--model",
            ]
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert output["operation"] == "models.unbind"
    assert output["result"] == {
        "application_id": "support-desk",
        "slot": "primary",
        "unbound": True,
    }


def test_a_refused_binding_reports_clouds_reason(cloud, capsys: pytest.CaptureFixture[str]) -> None:
    cloud(
        {
            ("GET", f"{_APPS}?lifecycle=all"): httpx.Response(200, json=_AGENTS),
            ("PUT", f"{_MODELS}/primary"): httpx.Response(
                409,
                json={
                    "detail": {
                        "code": "model_not_approved",
                        "message": "Approved models: bedrock/claude-sonnet-4-5.",
                    }
                },
            ),
        }
    )

    code = main(
        ["cloud", "connections", "bind", "primary", "--application", "support-desk", "--model", "x"]
    )
    output = json.loads(capsys.readouterr().out)
    assert code != 0
    assert output["error"]["code"] == "model_not_approved"
    assert "Approved models" in output["error"]["message"]


@pytest.mark.parametrize(
    ("status", "code"),
    [
        (409, "models_unavailable"),
        (503, "models_unavailable"),
        (503, "models_unreadable"),
        (422, "invalid_model"),
        (422, "invalid_name"),
    ],
)
def test_model_route_refusals_surface_their_code(
    cloud, capsys: pytest.CaptureFixture[str], status: int, code: str
) -> None:
    cloud(
        {
            ("GET", f"{_APPS}?lifecycle=all"): httpx.Response(200, json=_AGENTS),
            ("GET", _MODELS): httpx.Response(
                status, json={"detail": {"code": code, "message": "refused"}}
            ),
        }
    )
    assert main(["cloud", "models", "list", "--application", "support-desk", "--json"]) != 0
    assert json.loads(capsys.readouterr().out)["error"]["code"] == code


def test_a_model_binding_takes_no_connection_and_a_connection_binding_needs_one(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = cloud({("GET", f"{_APPS}?lifecycle=all"): httpx.Response(200, json=_AGENTS)})
    for arguments in (
        ["primary", "team-anthropic", "--model", "m"],
        ["primary", "--scope", "read", "--model", "m"],
        ["primary"],
    ):
        code = main(["cloud", "connections", "bind", *arguments, "--application", "support-desk"])
        assert code == 2
        assert json.loads(capsys.readouterr().out)["error"]["category"] == "invalid_input"
    assert all(request.method == "GET" for request in fake.requests)
