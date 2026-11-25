"""`cayu cloud ai-connections` calls Cloud's AI connection and AI setting routes."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from tests.cli.test_cloud_api_keys import cloud as _cloud

from cayu.cli import main

cloud = _cloud

_CONNECTIONS = "/v1/ai-connections"
_SETTINGS = "/v1/ai-settings"
_CONNECTION = {
    "id": "aic_1",
    "provider": "openai",
    "auth_type": "chatgpt_subscription",
    "status": "pending",
    "is_default": False,
}


def _run(arguments: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, dict[str, Any]]:
    code = main(["cloud", "ai-connections", *arguments])
    return code, json.loads(capsys.readouterr().out)


def _sent(fake: Any) -> list[tuple[str, str, object]]:
    return [
        (
            request.method,
            request.url.raw_path.decode(),
            json.loads(request.content) if request.content else None,
        )
        for request in fake.requests
    ]


def test_help_lists_every_command(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        main(["cloud", "ai-connections", "--help"])

    assert raised.value.code == 0
    help_text = capsys.readouterr().out
    for command in (
        "list",
        "connect",
        "usage",
        "settings",
        "use-cayu",
        "complete",
        "default",
        "disconnect",
        "use",
    ):
        assert command in help_text


@pytest.mark.parametrize(
    ("arguments", "path"),
    [
        (["list"], _CONNECTIONS),
        (["usage"], f"{_CONNECTIONS}/usage"),
        (["settings"], _SETTINGS),
    ],
)
def test_reads_call_their_route(
    cloud, capsys: pytest.CaptureFixture[str], arguments: list[str], path: str
) -> None:
    fake = cloud({("GET", path): httpx.Response(200, json={"items": []})})

    code, output = _run(arguments, capsys)

    assert code == 0, output
    assert output == {
        "ok": True,
        "operation": f"ai-connections.{arguments[0]}",
        "result": {"items": []},
    }
    assert _sent(fake) == [("GET", path, None)]


def test_connect_starts_a_device_sign_in_and_prints_its_code(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    start = {
        "connection": _CONNECTION,
        "user_code": "ABCD-1234",
        "verification_url": "https://auth.example/device",
        "poll_interval_seconds": 5,
    }
    fake = cloud({("POST", f"{_CONNECTIONS}/openai-subscription"): httpx.Response(201, json=start)})

    code, output = _run(["connect"], capsys)

    assert code == 0, output
    assert output["operation"] == "ai-connections.connect"
    assert output["result"] == start
    assert output["organization"]["id"] == "org_test"
    assert _sent(fake) == [("POST", f"{_CONNECTIONS}/openai-subscription", None)]


@pytest.mark.parametrize(
    ("action", "method", "path"),
    [
        ("complete", "POST", f"{_CONNECTIONS}/aic_1/complete"),
        ("default", "PUT", f"{_CONNECTIONS}/aic_1/default"),
    ],
)
def test_a_connection_change_calls_its_route(
    cloud, capsys: pytest.CaptureFixture[str], action: str, method: str, path: str
) -> None:
    ready = {**_CONNECTION, "status": "ready", "is_default": True}
    fake = cloud({(method, path): httpx.Response(200, json=ready)})

    code, output = _run([action, "aic_1"], capsys)

    assert code == 0, output
    assert (output["operation"], output["result"]) == (f"ai-connections.{action}", ready)
    assert _sent(fake) == [(method, path, None)]


def test_disconnect_says_which_connection_it_removed(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = cloud({("DELETE", f"{_CONNECTIONS}/aic_1"): httpx.Response(204)})

    code, output = _run(["disconnect", "aic_1"], capsys)

    assert code == 0, output
    assert output["result"] == {"id": "aic_1", "disconnected": True}
    assert _sent(fake) == [("DELETE", f"{_CONNECTIONS}/aic_1", None)]


def test_a_connection_id_is_sent_as_one_path_segment(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = cloud({("DELETE", f"{_CONNECTIONS}/a%2Fb"): httpx.Response(204)})

    code, _ = _run(["disconnect", "a/b"], capsys)

    assert code == 0
    assert _sent(fake)[0][1] == f"{_CONNECTIONS}/a%2Fb"


def test_use_chooses_a_connection_and_model(cloud, capsys: pytest.CaptureFixture[str]) -> None:
    chosen = {"configured": True, "mode": "byoai", "connection_id": "aic_1", "model": "gpt-5.4"}
    fake = cloud({("PUT", _SETTINGS): httpx.Response(200, json=chosen)})

    code, output = _run(["use", "aic_1", "--organization", "org_test"], capsys)
    assert code == 0, output
    code, output = _run(["use", "aic_1", "--model", "gpt-next"], capsys)
    assert code == 0, output

    assert [body for _, _, body in _sent(fake)] == [
        {"mode": "byoai", "connection_id": "aic_1", "model": "gpt-5.4"},
        {"mode": "byoai", "connection_id": "aic_1", "model": "gpt-next"},
    ]


def test_use_cayu_clears_the_connection_and_model(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = cloud({("PUT", _SETTINGS): httpx.Response(200, json={"mode": "cayu"})})

    code, output = _run(["use-cayu"], capsys)

    assert code == 0, output
    assert output["operation"] == "ai-connections.use-cayu"
    assert _sent(fake) == [
        ("PUT", _SETTINGS, {"mode": "cayu", "connection_id": None, "model": None})
    ]


def test_a_refused_change_reports_cloud_status(cloud, capsys: pytest.CaptureFixture[str]) -> None:
    cloud(
        {
            ("PUT", _SETTINGS): httpx.Response(
                409, json={"detail": "Choose a ready organization AI connection."}
            )
        }
    )

    code, output = _run(["use", "aic_1"], capsys)

    assert code != 0
    assert output["error"]["category"] == "api_request_rejected"
    assert "HTTP 409" in output["error"]["message"]


def test_a_change_outside_the_acting_organization_is_refused_before_anything_is_sent(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = cloud({})

    code, output = _run(["disconnect", "aic_1", "--organization", "org_other"], capsys)

    assert code != 0, output
    assert output["error"]["category"] == "organization_mismatch"
    assert fake.requests == []
