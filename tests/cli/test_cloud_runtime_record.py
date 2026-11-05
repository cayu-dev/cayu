"""`cayu cloud admin runtime-record show|forget-migration`."""

from __future__ import annotations

import argparse
import json
from typing import Any

import pytest

from cayu.cli import cloud as cloud_cli
from cayu.cli import main
from cayu.cli._cloud_api import CloudApiError, _safe_api_error_code
from cayu.cli._cloud_runtime_record import runtime_record_command

_PATH = "/v1/admin/organizations/org_acme/applications/scout-agent/runtime-record"
_FORGET = f"{_PATH}/forget-migration-notes"
_RECORD = {
    "deployment_id": "dep-two",
    "status": "running",
    "database": {"name": "cayu_agent_x", "migrated_image_digest": "sha256:" + "a" * 64},
    "app_database_migration": {
        "created_at": "2026-11-01T10:00:00+00:00",
        "database_name": "cayu_agent_x",
        "deployment_id": "dep-one",
    },
}
_SHOWN = {
    "organization_id": "org_acme",
    "application_id": "scout-agent",
    "revision": "17",
    "location": "cloud_database",
    "record": _RECORD,
}


class _Cloud:
    def __init__(self, answers: dict[str, dict[str, Any] | Exception]) -> None:
        self.answers = answers
        self.requests: list[tuple[str, str, object]] = []

    def request(self, method: str, path: str, **kwargs: object) -> dict[str, Any]:
        self.requests.append((method, path, kwargs.get("payload")))
        answer = self.answers[path]
        if isinstance(answer, Exception):
            raise answer
        return answer


def _run(monkeypatch, capsys, cloud: _Cloud, *argv: str) -> tuple[int, str]:
    monkeypatch.setattr(cloud_cli, "_cloud_client", lambda *_args, **_kwargs: cloud)
    code = main(["cloud", "admin", "runtime-record", *argv])
    return code, capsys.readouterr().out


def test_show_prints_the_whole_record_as_json(monkeypatch, capsys) -> None:
    cloud = _Cloud({_PATH: _SHOWN})

    code, out = _run(
        monkeypatch, capsys, cloud, "show", "scout-agent", "--organization", "org_acme", "--json"
    )

    assert code == 0
    assert json.loads(out) == {
        "ok": True,
        "operation": "admin.runtime-record.show",
        "result": {
            **_SHOWN,
            "commands": {
                "forget_migration": "cayu cloud admin runtime-record forget-migration "
                "scout-agent --organization org_acme --revision 17"
            },
        },
    }
    assert cloud.requests == [("GET", _PATH, None)]


def test_the_follow_up_command_keeps_the_selected_cloud() -> None:
    arguments = argparse.Namespace(
        runtime_record_command="show", application="scout-agent", organization="org_acme"
    )

    result = runtime_record_command(
        arguments,
        client=_Cloud({_PATH: _SHOWN}),  # type: ignore[arg-type]
        recovery_arguments=("--context", "staging.json"),
    )

    assert result["result"]["commands"]["forget_migration"] == (
        "cayu cloud --context staging.json admin runtime-record forget-migration scout-agent "
        "--organization org_acme --revision 17"
    )


def test_show_names_the_revision_and_the_notes_and_how_to_forget_one(monkeypatch, capsys) -> None:
    code, out = _run(
        monkeypatch,
        capsys,
        _Cloud({_PATH: _SHOWN}),
        "show",
        "scout-agent",
        "--organization",
        "org_acme",
    )

    assert code == 0
    assert "Revision:   17" in out
    assert "Kept in:    cloud_database" in out
    assert "app_database_migration  Release dep-one" in out
    assert "migrated_image_digest   sha256:" in out
    assert (
        "cayu cloud admin runtime-record forget-migration scout-agent --organization org_acme "
        "--revision 17" in out
    )


def test_control_characters_from_cloud_never_reach_the_terminal(monkeypatch, capsys) -> None:
    shown = {**_SHOWN, "revision": "17\x1b]0;owned\x07\nrm -rf /"}

    code, out = _run(
        monkeypatch,
        capsys,
        _Cloud({_PATH: shown}),
        "show",
        "scout-agent",
        "--organization",
        "org_acme",
    )

    assert code == 0
    assert "\x1b" not in out and "\x07" not in out
    assert "\nrm -rf" not in out


def test_a_record_without_notes_says_so(monkeypatch, capsys) -> None:
    shown = {**_SHOWN, "record": {"deployment_id": "dep-two", "status": "running"}}

    code, out = _run(
        monkeypatch,
        capsys,
        _Cloud({_PATH: shown}),
        "show",
        "scout-agent",
        "--organization",
        "org_acme",
    )

    assert code == 0
    assert "Migration notes:\n  none" in out
    assert "forget-migration" not in out


def test_forget_sends_the_revision_and_the_app_migration_note_by_default(
    monkeypatch, capsys
) -> None:
    written = {**_SHOWN, "revision": "18", "forgotten": ["app_database_migration"]}
    cloud = _Cloud({_FORGET: written})

    code, out = _run(
        monkeypatch,
        capsys,
        cloud,
        "forget-migration",
        "scout-agent",
        "--organization",
        "org_acme",
        "--revision",
        "17",
    )

    assert code == 0
    assert cloud.requests == [
        ("POST", _FORGET, {"expected_revision": "17", "notes": ["app_database_migration"]})
    ]
    assert out.strip() == (
        "Forgot app_database_migration. The record is at revision 18; deploy the Agent to run "
        "the migration again."
    )


def test_forget_names_both_notes_once_each(monkeypatch, capsys) -> None:
    written = {**_SHOWN, "forgotten": []}
    cloud = _Cloud({_FORGET: written})

    code, out = _run(
        monkeypatch,
        capsys,
        cloud,
        "forget-migration",
        "scout-agent",
        "--organization",
        "org_acme",
        "--revision",
        "0.0123456789abcdef",
        "--note",
        "migrated_image_digest",
        "--note",
        "app_database_migration",
        "--note",
        "migrated_image_digest",
    )

    assert code == 0
    assert cloud.requests[0][2] == {
        "expected_revision": "0.0123456789abcdef",
        "notes": ["app_database_migration", "migrated_image_digest"],
    }
    assert "nothing changed" in out


def test_an_organization_is_required_and_may_not_be_empty(monkeypatch, capsys) -> None:
    cloud = _Cloud({})
    code, _ = _run(monkeypatch, capsys, cloud, "show", "scout-agent")
    assert code == 2
    code, out = _run(
        monkeypatch, capsys, cloud, "show", "scout-agent", "--organization", " ", "--json"
    )
    assert code == 2
    assert json.loads(out)["error"]["category"] == "invalid_input"
    assert cloud.requests == []


@pytest.mark.parametrize(
    ("failure", "category"),
    [
        (
            CloudApiError("api_request_rejected", "HTTP 403.", status_code=403),
            "platform_admin_required",
        ),
        (
            CloudApiError("api_request_rejected", "HTTP 404.", status_code=404),
            "runtime_records_unsupported",
        ),
        (
            CloudApiError(
                "api_request_rejected",
                "The Agent's runtime record changed since it was read.",
                status_code=409,
                code="runtime_record_changed",
            ),
            "api_request_rejected",
        ),
    ],
)
def test_a_refusal_names_why(monkeypatch, capsys, failure: CloudApiError, category: str) -> None:
    code, out = _run(
        monkeypatch,
        capsys,
        _Cloud({_FORGET: failure}),
        "forget-migration",
        "scout-agent",
        "--organization",
        "org_acme",
        "--revision",
        "17",
        "--json",
    )

    assert code == 2
    assert json.loads(out)["error"]["category"] == category


@pytest.mark.parametrize(
    ("status", "code"),
    [
        (404, "application_not_found"),
        (404, "runtime_record_not_found"),
        (409, "runtime_record_changed"),
        (409, "database_migration_running"),
        (409, "service_busy"),
        (503, "application_runtime_unavailable"),
    ],
)
def test_cloud_s_refusals_reach_the_caller_with_their_code(status: int, code: str) -> None:
    assert _safe_api_error_code(status, (code, {"code": code, "message": "x"})) == code
