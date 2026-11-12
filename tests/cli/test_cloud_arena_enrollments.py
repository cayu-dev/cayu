"""`cayu cloud admin arena-enrollments [--below-schema N]`."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from cayu.cli import cloud as cloud_cli
from cayu.cli import main
from cayu.cli._cloud_api import CloudApiError, _rejection

_LIST = "/v1/admin/arena/enrollments"


def _row(
    agent_id: str,
    *,
    source: str,
    version: int | None,
    application_id: str | None = None,
    organization_id: str | None = "org_acme",
    reason: str | None = None,
    revision: str | None = None,
    corpora: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "agent_id": agent_id,
        "organization_id": organization_id,
        "application_id": application_id,
        "bound": application_id is not None,
        "enrollment": {
            "sequence": 1,
            "fingerprint": "f" * 64,
            "declaration_fingerprint": "sha256:" + "d" * 64,
            "enrolled_by": "alice@example.com",
            "enrolled_at": "2026-10-26T17:00:00+00:00",
            "source_revision": revision,
        },
        "arena_error": None,
        "corpus_schema": {
            "source": source,
            "version": version,
            "corpora": corpora
            if corpora is not None
            else ([] if version is None else [{"path": "evals/c.json", "schema_version": version}]),
            "release": None,
            "reason": reason,
        },
    }


_ITEMS = [
    _row("finance-monthend", source="current_release", version=3, application_id="finance"),
    _row(
        "invoices",
        source="recorded",
        version=6,
        application_id="invoices",
        revision="4263a40" + "0" * 33,
    ),
    _row(
        "partial",
        source="current_release",
        version=5,
        application_id="partial",
        corpora=[
            {"path": "evals/a.json", "schema_version": 5, "problem": None},
            {"path": "evals/b.json", "schema_version": None, "problem": "missing"},
        ],
    ),
    _row("unbound", source="unknown", version=None, organization_id=None, reason="not_bound"),
]


class _Cloud:
    def __init__(self, answer: dict[str, Any] | Exception) -> None:
        self.answer = answer
        self.requests: list[tuple[str, str]] = []

    def request(self, method: str, path: str, **kwargs: object) -> dict[str, Any]:
        del kwargs
        self.requests.append((method, path))
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


def _run(monkeypatch, capsys, cloud: _Cloud, *argv: str) -> tuple[int, str]:
    monkeypatch.setattr(cloud_cli, "_cloud_client", lambda *_args, **_kwargs: cloud)
    code = main(["cloud", "admin", "arena-enrollments", *argv])
    return code, capsys.readouterr().out


def test_json_lists_every_enrollment(monkeypatch, capsys) -> None:
    cloud = _Cloud({"items": _ITEMS})

    code, out = _run(monkeypatch, capsys, cloud, "--json")

    assert code == 0
    assert json.loads(out) == {
        "ok": True,
        "operation": "admin.arena-enrollments",
        "result": {"items": _ITEMS},
    }
    assert cloud.requests == [("GET", _LIST)]


def test_below_schema_keeps_the_lower_schemas_with_their_source(monkeypatch, capsys) -> None:
    code, out = _run(
        monkeypatch, capsys, _Cloud({"items": _ITEMS}), "--below-schema", "6", "--json"
    )

    assert code == 0
    result = json.loads(out)["result"]
    assert [item["agent_id"] for item in result["items"]] == ["finance-monthend", "partial"]
    assert [item["corpus_schema"]["source"] for item in result["items"]] == [
        "current_release",
        "current_release",
    ]
    assert result["below_schema"] == 6
    assert result["without_schema"] == 1


def test_the_table_shows_each_schema_and_where_it_came_from(monkeypatch, capsys) -> None:
    code, out = _run(monkeypatch, capsys, _Cloud({"items": _ITEMS}))

    assert code == 0
    lines = out.splitlines()
    assert lines[0].split() == [
        "ORGANIZATION",
        "AGENT",
        "CLOUD",
        "AGENT",
        "REVISION",
        "ENROLLMENT",
        "SCHEMA",
        "SOURCE",
    ]
    rows = {line.split()[1]: line for line in lines[1:5]}
    assert rows["finance-monthend"].split()[-2:] == ["3", "current_release"]
    assert "4263a4000000" in rows["invoices"]
    assert rows["invoices"].split()[-2:] == ["6", "recorded"]
    assert "5 (some unread)" in rows["partial"]
    assert "unbound" in rows["unbound"]
    assert rows["unbound"].rstrip().endswith("unknown (not_bound)")
    assert "#1 ffffffffffff" in rows["invoices"]
    assert "recorded = Arena recorded it at enrollment" in out


def test_below_schema_says_what_it_leaves_out(monkeypatch, capsys) -> None:
    code, out = _run(monkeypatch, capsys, _Cloud({"items": _ITEMS}), "--below-schema", "3")

    assert code == 0
    assert "No enrollment shows a corpus schema below 3." in out
    assert "1 enrollment shows no corpus schema" in out
    assert "isn't listed" in out


def test_an_install_without_enrollments_says_so(monkeypatch, capsys) -> None:
    code, out = _run(monkeypatch, capsys, _Cloud({"items": []}))

    assert code == 0
    assert out.strip() == "Arena has no case enrollments."


@pytest.mark.parametrize("value", ["0", "-1", "six", "6.0"])
def test_below_schema_takes_a_positive_whole_number(monkeypatch, capsys, value: str) -> None:
    cloud = _Cloud({"items": []})

    code, out = _run(monkeypatch, capsys, cloud, "--below-schema", value)

    assert code == 2
    error = json.loads(out)["error"]
    assert error["category"] == "invalid_input"
    assert "--below-schema" in error["message"]
    assert cloud.requests == []


@pytest.mark.parametrize(
    ("failure", "category"),
    [
        (
            CloudApiError("api_request_rejected", "HTTP 403.", status_code=403),
            "platform_admin_required",
        ),
        (CloudApiError(None, "HTTP 404.", status_code=404), "arena_enrollments_unsupported"),
    ],
)
def test_a_refusal_names_why(monkeypatch, capsys, failure: CloudApiError, category: str) -> None:
    code, out = _run(monkeypatch, capsys, _Cloud(failure), "--json")

    assert code == 2
    assert json.loads(out)["error"]["category"] == category


@pytest.mark.parametrize(
    ("status", "code"),
    [
        (503, "arena_inventory_not_configured"),
        (503, "arena_inventory_unconfigured"),
        (502, "arena_inventory_refused"),
        (503, "arena_inventory_unsupported"),
        (503, "arena_inventory_unavailable"),
    ],
)
def test_clouds_refusal_names_what_to_set(monkeypatch, capsys, status: int, code: str) -> None:
    message = "Set CAYU_CLOUD_ARENA_INVENTORY_TOKEN, then restart the API."
    refusal = _rejection(
        httpx.Response(status, json={"detail": {"code": code, "message": message}})
    )

    exit_code, out = _run(monkeypatch, capsys, _Cloud(refusal), "--json")

    assert exit_code == 2
    error = json.loads(out)["error"]
    assert (error["category"], error["code"]) == (code, code)
    assert message in error["message"]


def test_an_invalid_answer_is_refused(monkeypatch, capsys) -> None:
    code, out = _run(monkeypatch, capsys, _Cloud({"items": "x"}), "--json")

    assert code == 2
    assert json.loads(out)["error"]["category"] == "api_response_invalid"
