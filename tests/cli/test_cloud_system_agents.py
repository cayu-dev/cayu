"""`cayu cloud admin system-agents`, and what `service status` says when the install's system
Agent setting changed after the service was published."""

from __future__ import annotations

import json
import shlex
from typing import Any

import pytest

from cayu.cli import cloud as cloud_cli
from cayu.cli import main
from cayu.cli._cloud_api import CloudApiError
from cayu.cli._cloud_system_agents import system_change_warning

_LIST = "/v1/admin/system-agents"
_ITEMS = [
    {
        "entry": "org_cayu/compound",
        "organization_id": "org_cayu",
        "agent_id": "compound",
        "setting": "CAYU_SYSTEM_AGENTS",
        "deprecated": False,
        "agent": {"organization_id": "org_cayu", "lifecycle": "active", "recorded_system": True},
    },
    {
        "entry": "org_cayu/cayu-compass",
        "organization_id": "org_cayu",
        "agent_id": "cayu-compass",
        "setting": "CAYU_SYSTEM_AGENTS",
        "deprecated": False,
        "agent": {"organization_id": "org_cayu", "lifecycle": "active", "recorded_system": False},
    },
    {
        "entry": "org_cayu/misspelt",
        "organization_id": "org_cayu",
        "agent_id": "misspelt",
        "setting": "CAYU_SYSTEM_AGENTS",
        "deprecated": False,
        "agent": None,
    },
    {
        "entry": "cayu-arena-cases",
        "organization_id": None,
        "agent_id": "cayu-arena-cases",
        "setting": "CAYU_AGENT_ALWAYS_ON",
        "deprecated": True,
        "agent": {"organization_id": "org_arena", "lifecycle": "active", "recorded_system": None},
    },
]


class _Cloud:
    def __init__(self, answers: dict[str, dict[str, Any] | Exception]) -> None:
        self.answers = answers
        self.requests: list[tuple[str, str]] = []

    def request(self, method: str, path: str, **kwargs: object) -> dict[str, Any]:
        del kwargs
        self.requests.append((method, path))
        answer = self.answers[path]
        if isinstance(answer, Exception):
            raise answer
        return answer


def _run(monkeypatch, capsys, cloud: _Cloud, *argv: str) -> tuple[int, str]:
    monkeypatch.setattr(cloud_cli, "_cloud_client", lambda *_args, **_kwargs: cloud)
    code = main(["cloud", *argv])
    return code, capsys.readouterr().out


def test_json_lists_each_entry_and_the_agent_it_matches(monkeypatch, capsys) -> None:
    cloud = _Cloud({_LIST: {"items": _ITEMS}})

    code, out = _run(monkeypatch, capsys, cloud, "admin", "system-agents", "--json")

    assert code == 0
    assert json.loads(out) == {
        "ok": True,
        "operation": "admin.system-agents",
        "result": {"items": _ITEMS},
    }
    assert cloud.requests == [("GET", _LIST)]


def test_the_table_marks_deprecated_entries_wrong_ones_and_services_to_restart(
    monkeypatch, capsys
) -> None:
    code, out = _run(
        monkeypatch, capsys, _Cloud({_LIST: {"items": _ITEMS}}), "admin", "system-agents"
    )

    assert code == 0
    lines = out.splitlines()
    assert lines[0].split() == ["ENTRY", "SETTING", "ORGANIZATION", "LIFECYCLE", "SERVICE"]
    rows = {line.split()[0]: line for line in lines[1:5]}
    assert rows["org_cayu/compound"].split()[1:] == [
        "CAYU_SYSTEM_AGENTS",
        "org_cayu",
        "active",
        "system",
    ]
    assert "not system until restarted" in rows["org_cayu/cayu-compass"]
    assert "no such Agent" in rows["org_cayu/misspelt"]
    assert "CAYU_AGENT_ALWAYS_ON (deprecated)" in rows["cayu-arena-cases"]
    assert "not recorded" in rows["cayu-arena-cases"]
    assert "Recorded as not a system Agent: org_cayu/cayu-compass." in out
    assert "check its ids" in out


def test_an_install_without_system_agents_says_so(monkeypatch, capsys) -> None:
    code, out = _run(monkeypatch, capsys, _Cloud({_LIST: {"items": []}}), "admin", "system-agents")

    assert code == 0
    assert out.strip() == "This install names no system Agents."


@pytest.mark.parametrize(
    ("failure", "category"),
    [
        (
            CloudApiError("api_request_rejected", "HTTP 403.", status_code=403),
            "platform_admin_required",
        ),
        (
            CloudApiError("api_request_rejected", "HTTP 404.", status_code=404),
            "system_agents_unsupported",
        ),
    ],
)
def test_a_refusal_names_why(monkeypatch, capsys, failure: CloudApiError, category: str) -> None:
    code, out = _run(
        monkeypatch, capsys, _Cloud({_LIST: failure}), "admin", "system-agents", "--json"
    )

    assert code == 2
    assert json.loads(out)["error"]["category"] == category


def test_an_invalid_answer_is_refused(monkeypatch, capsys) -> None:
    code, out = _run(
        monkeypatch, capsys, _Cloud({_LIST: {"items": "x"}}), "admin", "system-agents", "--json"
    )

    assert code == 2
    assert json.loads(out)["error"]["category"] == "api_response_invalid"


def _status(monkeypatch, capsys, service: dict[str, Any]) -> dict[str, Any]:
    cloud = _Cloud(
        {
            "/v1/applications": {"items": [{"id": "compound", "name": "Compound"}]},
            "/v1/applications/compound/service": service,
            "/v1/applications/compound/worker-exits": {"items": []},
        }
    )
    code, out = _run(monkeypatch, capsys, cloud, "service", "status", "--application", "compound")
    assert code == 0
    return json.loads(out)


def test_service_status_warns_when_the_recorded_value_differs(monkeypatch, capsys) -> None:
    service = {
        "application_id": "compound",
        "deployment_id": "dep_1",
        "status": "running",
        "system": True,
        "recorded_system": False,
    }

    output = _status(monkeypatch, capsys, service)

    (warning,) = output["warnings"]
    assert "now names this Agent one of its system Agents" in warning
    assert "`cayu cloud service restart --application compound`" in warning
    assert output["result"]["system"] is True
    assert output["result"]["recorded_system"] is False


@pytest.mark.parametrize(
    "fields",
    [
        {"system": True, "recorded_system": True},
        # A Cloud that doesn't report the recorded value reports it as `system`.
        {"system": False},
    ],
)
def test_service_status_has_no_warning_when_they_agree(
    monkeypatch, capsys, fields: dict[str, Any]
) -> None:
    service = {"application_id": "compound", "deployment_id": "dep_1", "status": "running"}

    output = _status(monkeypatch, capsys, {**service, **fields})

    assert "warnings" not in output


def test_the_restart_it_names_keeps_the_selected_cloud(monkeypatch, capsys, tmp_path) -> None:
    context = tmp_path / "staging cloud.json"
    key_file = tmp_path / "key"
    key_file.write_text("cayu_test\n")
    context.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "ready",
                "api_url": "https://staging.example.test",
                "api_key_file": str(key_file),
            }
        )
    )
    cloud = _Cloud(
        {
            "/v1/applications": {"items": [{"id": "compound", "name": "Compound"}]},
            "/v1/applications/compound/service": {
                "application_id": "compound",
                "deployment_id": "dep_1",
                "status": "running",
                "system": True,
                "recorded_system": False,
            },
            "/v1/applications/compound/worker-exits": {"items": []},
        }
    )
    argv = ["--context", str(context), "--api-key-file", str(key_file)]

    code, out = _run(
        monkeypatch, capsys, cloud, *argv, "service", "status", "--application", "compound"
    )

    assert code == 0, out
    (warning,) = json.loads(out)["warnings"]
    restart = shlex.join(
        ["cayu", "cloud", *argv, "service", "restart", "--application", "compound"]
    )
    assert f"`{restart}`" in warning


def test_the_warning_names_an_agent_no_longer_listed() -> None:
    warning = system_change_warning(
        {"system": False, "recorded_system": True}, application="support-desk"
    )

    assert warning is not None
    assert warning.startswith("The install no longer names this Agent")
    assert (
        system_change_warning({"system": False, "recorded_system": None}, application="x") is None
    )
