"""`cayu cloud applications export|import`: copy an Agent's database and files."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from cayu.cli import cloud as cloud_cli
from cayu.cli._cloud_api import CloudApiError, _rejection

EXPORT_ID = "exp_0123456789abcdef01234567"
IMPORT_ID = "imp_0123456789abcdef01234567"


class TransferClient:
    """Cayu Cloud's transfer routes: each one finishes after ``pending`` status reads."""

    def __init__(self, *, pending: int = 1, outcome: str = "succeeded") -> None:
        self.pending = pending
        self.outcome = outcome
        self.requests: list[tuple[str, str, dict[str, object]]] = []
        self.failures: list[CloudApiError] = []

    def record(self, kind: str, status: str) -> dict[str, Any]:
        transfer: dict[str, Any] = {
            "id": EXPORT_ID if kind == "export" else IMPORT_ID,
            "kind": kind,
            "status": status,
            "error": None,
        }
        if status == "failed":
            transfer["error"] = {
                "code": "target_has_data",
                "message": "The target Agent's database already holds data (cayu_sessions).",
            }
        if kind == "import" and status == "succeeded":
            transfer["missing_secrets"] = ["OPENAI_API_KEY"]
        return transfer

    def request(self, method: str, path: str, **kwargs: object) -> dict[str, Any]:
        self.requests.append((method, path, kwargs))
        kind = "import" if "/data-imports" in path else "export"
        if method == "GET" and path.endswith(("/data-exports", "/data-imports")):
            return {"items": [self.record(kind, "succeeded")]}
        if method == "POST":
            return self.record(kind, "queued")
        if self.failures:
            raise self.failures.pop(0)
        if self.pending > 0:
            self.pending -= 1
            return self.record(kind, "running")
        return self.record(kind, self.outcome)


def parse(*argv: str):
    return cloud_cli._build_parser().parse_args(["applications", *argv])


def run(arguments, client: TransferClient, *, clock: list[float] | None = None) -> dict[str, Any]:
    ticks = iter(clock or [0.0] * 50)
    return cloud_cli._applications(
        arguments, client=client, sleep=lambda _: None, monotonic=lambda: next(ticks)
    )


def test_export_asks_cloud_and_waits_until_the_archive_is_stored() -> None:
    client = TransferClient(pending=2)

    result = run(parse("export", "support-desk"), client)

    assert result == {
        "operation": "applications.export",
        "result": {"id": EXPORT_ID, "kind": "export", "status": "succeeded", "error": None},
    }
    assert client.requests[0] == (
        "POST",
        "/v1/applications/support-desk/data-exports",
        {"payload": None},
    )
    assert [request[:2] for request in client.requests[1:]] == [
        ("GET", f"/v1/applications/support-desk/data-exports/{EXPORT_ID}")
    ] * 3


def test_import_names_the_export_and_reports_the_secrets_still_needed() -> None:
    client = TransferClient(pending=0)

    result = run(parse("import", "support-copy", "--from", EXPORT_ID, "--overwrite"), client)

    assert result["operation"] == "applications.import"
    assert result["result"]["missing_secrets"] == ["OPENAI_API_KEY"]
    assert client.requests[0] == (
        "POST",
        "/v1/applications/support-copy/data-imports",
        {"payload": {"export_id": EXPORT_ID, "overwrite": True}},
    )


def test_a_platform_admin_names_the_organization_and_no_wait_returns_once_queued() -> None:
    client = TransferClient()

    result = run(
        parse(
            "import", "support-copy", "--from", EXPORT_ID, "--organization", "org_beta", "--no-wait"
        ),
        client,
    )

    assert result["result"]["status"] == "queued"
    assert client.requests == [
        (
            "POST",
            "/v1/admin/organizations/org_beta/applications/support-copy/data-imports",
            {"payload": {"export_id": EXPORT_ID, "overwrite": False}},
        )
    ]


def test_listing_and_reading_one_transfer() -> None:
    client = TransferClient(pending=0)

    listed = run(parse("exports", "support-desk"), client)
    shown = run(
        parse("import-status", "support-copy", IMPORT_ID, "--organization", "org_a"), client
    )

    assert listed == {
        "operation": "applications.exports",
        "result": {
            "items": [{"id": EXPORT_ID, "kind": "export", "status": "succeeded", "error": None}]
        },
    }
    assert shown["result"]["id"] == IMPORT_ID
    assert [request[:2] for request in client.requests] == [
        ("GET", "/v1/applications/support-desk/data-exports"),
        (
            "GET",
            f"/v1/admin/organizations/org_a/applications/support-copy/data-imports/{IMPORT_ID}",
        ),
    ]


def test_a_failed_import_exits_2_with_clouds_code_and_message(
    capsys: pytest.CaptureFixture[str],
) -> None:
    client = TransferClient(pending=0, outcome="failed")

    with pytest.raises(CloudApiError) as raised:
        run(parse("import", "support-copy", "--from", EXPORT_ID), client)

    assert cloud_cli._cloud_failure(raised.value) == 2
    error = json.loads(capsys.readouterr().out)["error"]
    assert (error["category"], error["code"], error["id"]) == (
        "agent_data_import_failed",
        "target_has_data",
        IMPORT_ID,
    )
    assert "cayu_sessions" in error["message"]
    assert error["transfer"]["status"] == "failed"


def test_a_wait_that_runs_out_names_the_command_that_reads_it(
    capsys: pytest.CaptureFixture[str],
) -> None:
    client = TransferClient(pending=10)
    arguments = parse("export", "support-desk", "--wait-seconds", "2", "--poll-seconds", "1")

    with pytest.raises(CloudApiError) as raised:
        run(arguments, client, clock=[0.0, 1.0, 3.0])

    assert cloud_cli._cloud_failure(raised.value) == 2
    error = json.loads(capsys.readouterr().out)["error"]
    assert error == {
        "category": "agent_data_transfer_still_running",
        "commands": {"status": f"cayu cloud applications export-status support-desk {EXPORT_ID}"},
        "id": EXPORT_ID,
        "message": "Cayu Cloud is still running this export; it continues without the CLI.",
        "status": "running",
    }


def test_brief_outages_while_waiting_are_retried() -> None:
    client = TransferClient(pending=0)
    client.failures = [
        CloudApiError("api_unavailable", "Cayu Cloud is unavailable."),
        CloudApiError("api_request_rejected", "HTTP 503.", status_code=503),
    ]

    result = run(parse("export", "support-desk"), client)

    assert result["result"]["status"] == "succeeded"


def test_an_empty_organization_is_refused_rather_than_meaning_the_selected_one() -> None:
    client = TransferClient()

    with pytest.raises(CloudApiError) as refused:
        run(
            parse(
                "import", "support-copy", "--from", EXPORT_ID, "--organization", "", "--overwrite"
            ),
            client,
        )

    assert refused.value.category == "invalid_input"
    assert client.requests == []


def test_an_export_needs_an_exact_slug_and_an_import_its_export() -> None:
    with pytest.raises(cloud_cli.CloudCommandError):
        parse("export", "Support Desk")
    with pytest.raises(cloud_cli.CloudCommandError, match="--from"):
        parse("import", "support-copy")


def test_clouds_refusals_are_reported_with_their_code() -> None:
    refused = _rejection(
        httpx.Response(
            409,
            json={
                "detail": {
                    "code": "target_not_stopped",
                    "message": "The target Agent's service is running. Stop it first.",
                }
            },
        )
    )

    assert refused.code == "target_not_stopped"
    assert "Stop it first" in str(refused)
