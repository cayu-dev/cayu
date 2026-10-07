from __future__ import annotations

import copy
import json
import shlex
from typing import Any

import pytest

from cayu.cli import cloud
from cayu.cli._cloud_api import CloudApiError
from cayu.cli._cloud_diagnostics import parse_build_failure


def failure() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "code": "new_build_error",
        "phase": "image_built",
        "message": "The build could not finish.",
        "detail": "An unfamiliar compiler rejected the selected configuration.",
        "hint": "Correct the selected build configuration and deploy new source.",
        "automatic_retryable": True,
        "attempt": 2,
        "diagnostic_ref": "image_built:attempt-2",
        "diagnostic": {
            "status": "available",
            "stage": "docker_build",
            "exit_code": 2,
            "reason": None,
            "excerpt": "error: unsupported configuration frobnicate\nerror: compilation failed",
            "truncated": True,
        },
    }


class Client:
    def __init__(self, diagnostic: object) -> None:
        self.diagnostic = diagnostic
        self.requests: list[str] = []

    def request(self, method: str, path: str, **_: object) -> dict[str, Any]:
        self.requests.append(path)
        if path.endswith("/timeline"):
            return {"failure": self.diagnostic}
        return {"status": "failed", "id": "dep_failed"}


def wait_error(client: Client) -> CloudApiError:
    with pytest.raises(CloudApiError) as raised:
        cloud._wait_for_deployment(
            client,
            application_id="example-agent",
            deployment_id="dep_failed",
            include_failure_diagnostics=True,
            poll_seconds=0.01,
            wait_seconds=1,
            sleep=lambda _: None,
            monotonic=lambda: 0.0,
        )
    return raised.value


def test_new_safe_server_wording_and_code_survive_json_output(
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = failure()
    client = Client(payload)
    error = wait_error(client)
    assert cloud._cloud_failure(error) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["error"]["category"] == "new_build_error"
    assert result["error"]["failure"] == payload
    assert len(client.requests) == 2


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", 2),
        ("schema_version", True),
        ("automatic_retryable", "yes"),
        ("detail", "x" * 4097),
        ("hint", "API_KEY=credential-canary"),
        ("message", "error: sk-proj-credential-canary"),
        ("message", "error: before\x1b[31mafter"),
        ("message", "error: before\u202eafter"),
        ("message", "error: \ud800"),
        ("attempt", True),
        ("diagnostic_ref", "https://provider.invalid/private"),
        ("code", "provider/code"),
        ("diagnostic_ref", "smoke_tested:attempt-2"),
    ],
)
def test_malformed_or_unsafe_contract_has_safe_actionable_fallback(
    field: str,
    value: object,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = failure()
    payload[field] = value
    assert parse_build_failure(payload) is None
    error = wait_error(Client(payload))
    assert cloud._cloud_failure(error) == 2
    rendered = capsys.readouterr().out
    result = json.loads(rendered)["error"]
    assert result["category"] == "deployment_failed"
    assert result["diagnostic_status"] == "unavailable_or_unsupported"
    assert result["deployment_id"] == "dep_failed"
    assert (
        result["commands"]["logs"]
        == "cayu cloud deployment logs dep_failed --application example-agent"
    )
    assert "credential-canary" not in rendered
    assert "provider.invalid" not in rendered


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", []),
        ("exit_code", True),
        ("exit_code", 256),
        ("truncated", "yes"),
        ("excerpt", "password: credential-canary"),
        ("excerpt", "error\x00injection"),
        ("excerpt", "x" * 4097),
        ("reason", "https://provider.invalid"),
    ],
)
def test_invalid_evidence_is_not_published(field: str, value: object) -> None:
    payload = failure()
    payload["diagnostic"][field] = value
    assert parse_build_failure(payload) is None
    result = cloud._validated_deployment_diagnostics(
        {"diagnostics": [payload], "next_diagnostic_offset": None}
    )
    assert result["diagnostics"] == []
    assert result["diagnostic_status"] == "unavailable_or_unsupported"


def test_unavailable_evidence_preserves_original_failure() -> None:
    payload = failure()
    payload["diagnostic"].update(
        status="unavailable", reason="read_failed", excerpt="", exit_code=None
    )
    assert parse_build_failure(payload) == payload


def test_supported_log_page_keeps_only_contract_fields() -> None:
    payload = failure()
    expected = copy.deepcopy(payload)
    payload["provider_debug"] = "private-canary"
    payload["diagnostic"]["provider_url"] = "https://provider.invalid"
    result = cloud._validated_deployment_diagnostics(
        {"diagnostics": [payload], "next_diagnostic_offset": 20}
    )
    assert result == {"diagnostics": [expected], "next_diagnostic_offset": 20}


def test_logs_pagination_is_noninteractive() -> None:
    parser = cloud._build_parser()
    args = parser.parse_args(
        [
            "deployment",
            "logs",
            "dep_failed",
            "--application",
            "example-agent",
            "--diagnostic-offset",
            "20",
            "--diagnostic-limit",
            "5",
        ]
    )

    class LogClient:
        def request(self, method: str, path: str, **kwargs: object) -> dict[str, Any]:
            if path == "/v1/applications":
                return {"items": [{"id": "example-agent", "name": "Example"}]}
            assert path.endswith("/logs")
            assert kwargs["query"] == {"diagnostic_offset": "20", "diagnostic_limit": "5"}
            return {"diagnostics": [failure()], "next_diagnostic_offset": None}

    result = cloud._deployment(args, client=LogClient())
    assert result["result"]["diagnostics"] == [failure()]


@pytest.mark.parametrize("status", ["available", "unavailable", "withheld"])
def test_valid_failure_retains_recovery_command_for_explicit_logs(
    status: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = failure()
    if status != "available":
        payload["diagnostic"].update(
            status=status, reason="read_failed", excerpt="", exit_code=None
        )
    with pytest.raises(CloudApiError) as raised:
        cloud._wait_for_deployment(
            Client(payload),
            application_id="example-agent",
            deployment_id="dep_failed",
            include_failure_diagnostics=True,
            recovery_arguments=("--context", "/tmp/cloud context.json"),
            poll_seconds=0.01,
            wait_seconds=1,
            sleep=lambda _: None,
            monotonic=lambda: 0.0,
        )
    assert cloud._cloud_failure(raised.value) == 2
    error = json.loads(capsys.readouterr().out)["error"]
    assert error["failure"] == payload
    assert error["application"] == "example-agent"
    assert error["deployment_id"] == "dep_failed"
    command = shlex.split(error["commands"]["logs"])
    assert command[:4] == ["cayu", "cloud", "--context", "/tmp/cloud context.json"]
    arguments = cloud._build_parser().parse_args(command[2:])

    class LogClient:
        def request(self, method: str, path: str, **_: object) -> dict[str, Any]:
            if path == "/v1/applications":
                return {"items": [{"id": "example-agent", "name": "Example"}]}
            assert path == "/v1/applications/example-agent/deployments/dep_failed/logs"
            return {"diagnostics": [payload], "next_diagnostic_offset": None}

    result = cloud._deployment(arguments, client=LogClient())
    assert result["result"]["diagnostics"] == [payload]


@pytest.mark.parametrize(
    "phase,stage",
    [("database_provisioned", "image_build"), ("database_migrated", "database_migration")],
)
def test_cloud_database_publication_failures_parse_as_versioned_diagnostics(
    phase: str, stage: str
) -> None:
    payload = failure()
    payload.update(phase=phase, attempt=None, diagnostic_ref=None)
    payload["diagnostic"] = {
        "status": "unavailable",
        "stage": stage,
        "exit_code": None,
        "reason": "not_recorded",
        "excerpt": "",
        "truncated": False,
    }
    assert parse_build_failure(payload) == payload
    assert wait_error(Client(payload)).failure == payload


def smoke_failure(stage: object) -> dict[str, Any]:
    """Cloud's release smoke-test failure, as Cloud's timeline reports it."""

    return {
        "schema_version": 1,
        "code": "release_smoke_failed",
        "phase": "smoke_tested",
        "message": "The release smoke test failed.",
        "detail": "Cloud's credential-free check failed inside the release sandbox.",
        "hint": "Inspect the deployment diagnostics. Provider checks are Cloud-owned; do not "
        "add unrelated provider credentials to the Agent.",
        "automatic_retryable": True,
        "attempt": None,
        "diagnostic_ref": None,
        "diagnostic": {
            "status": "unavailable",
            "stage": stage,
            "exit_code": None,
            "reason": "not_applicable",
            "excerpt": "",
            "truncated": False,
        },
    }


def test_smoke_stage_failure_keeps_message_hint_and_stage(
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = smoke_failure("smoke_test")
    assert parse_build_failure(payload) == payload
    error = wait_error(Client(payload))
    assert cloud._cloud_failure(error) == 2
    result = json.loads(capsys.readouterr().out)["error"]
    assert result["category"] == "release_smoke_failed"
    assert result["message"] == "The release smoke test failed."
    assert result["failure"] == payload
    assert result["failure"]["diagnostic"]["stage"] == "smoke_test"
    assert "diagnostic_status" not in result
    assert set(result["commands"]) == {"logs", "timeline", "retry"}


@pytest.mark.parametrize(
    "stage,shown",
    [
        ("release_verification", "release_verification"),
        ("Smoke Test", "unknown"),
        ("https://provider.invalid/stage", "unknown"),
        (None, "unknown"),
        ([], "unknown"),
    ],
)
def test_unknown_stage_keeps_the_failure(stage: object, shown: str) -> None:
    payload = smoke_failure(stage)
    expected = copy.deepcopy(payload)
    expected["diagnostic"]["stage"] = shown
    assert parse_build_failure(payload) == expected
    assert wait_error(Client(payload)).failure == expected
    result = cloud._validated_deployment_diagnostics(
        {"diagnostics": [payload], "next_diagnostic_offset": None}
    )
    assert result == {"diagnostics": [expected], "next_diagnostic_offset": None}


def test_missing_stage_is_reported_as_unknown() -> None:
    payload = smoke_failure("smoke_test")
    del payload["diagnostic"]["stage"]
    parsed = parse_build_failure(payload)
    assert parsed is not None
    assert parsed["diagnostic"]["stage"] == "unknown"
    assert parsed["hint"] == payload["hint"]


@pytest.mark.parametrize("field", ["phase", "stage"])
@pytest.mark.parametrize(
    "private_value",
    [
        "ghp_credentialcanary123456",
        "github_pat_credentialcanary123456",
        "sk_credentialcanary123456",
        "cloudwatch",
        "a" * 64,
    ],
)
def test_private_identifiers_are_not_published(
    field: str, private_value: str, capsys: pytest.CaptureFixture[str]
) -> None:
    payload = failure()
    expected = copy.deepcopy(payload)
    if field == "phase":
        payload.update(phase=private_value, diagnostic_ref=f"{private_value}:attempt-2")
        expected.update(phase="unknown", diagnostic_ref=None)
    else:
        payload["diagnostic"]["stage"] = private_value
        expected["diagnostic"]["stage"] = "unknown"

    error = wait_error(Client(payload))
    assert cloud._cloud_failure(error) == 2
    rendered = capsys.readouterr().out
    result = json.loads(rendered)["error"]
    assert result["failure"] == expected
    assert result["category"] == expected["code"]
    assert result["message"] == expected["message"]
    assert private_value not in rendered

    diagnostics = cloud._validated_deployment_diagnostics(
        {"failure": payload, "diagnostics": [payload], "next_diagnostic_offset": None}
    )
    assert diagnostics == {
        "failure": expected,
        "diagnostics": [expected],
        "next_diagnostic_offset": None,
    }
    assert private_value not in json.dumps(diagnostics)


@pytest.mark.parametrize(
    "stage", ["source_validation", "docker_build", "image_build", "database_migration"]
)
def test_existing_stages_are_unchanged(stage: str) -> None:
    payload = failure()
    payload["diagnostic"]["stage"] = stage
    assert parse_build_failure(payload) == payload


@pytest.mark.parametrize(
    "phase,shown",
    [
        ("release_verified", "release_verified"),
        ("Image Built", "unknown"),
        ("https://provider.invalid/phase", "unknown"),
        (None, "unknown"),
        ([], "unknown"),
    ],
)
def test_unknown_phase_keeps_the_failure(
    phase: object, shown: str, capsys: pytest.CaptureFixture[str]
) -> None:
    payload = failure()
    payload["phase"] = phase
    payload["diagnostic_ref"] = f"{phase}:attempt-2" if shown == phase else None
    expected = copy.deepcopy(payload)
    expected["phase"] = shown
    assert parse_build_failure(payload) == expected
    error = wait_error(Client(payload))
    assert cloud._cloud_failure(error) == 2
    rendered = capsys.readouterr().out
    result = json.loads(rendered)["error"]
    assert result["category"] == "new_build_error"
    assert result["message"] == payload["message"]
    assert result["failure"] == expected
    assert "diagnostic_status" not in result
    assert "provider.invalid" not in rendered


def test_missing_phase_is_unknown_and_drops_its_unverifiable_reference() -> None:
    payload = failure()
    del payload["phase"]
    parsed = parse_build_failure(payload)
    assert parsed is not None
    assert parsed["phase"] == "unknown"
    assert parsed["diagnostic_ref"] is None
    assert (parsed["code"], parsed["message"], parsed["hint"]) == (
        payload["code"],
        payload["message"],
        payload["hint"],
    )


@pytest.mark.parametrize(
    "phase",
    [
        "source_resolved",
        "image_built",
        "image_scanned",
        "sandbox_template_ready",
        "policy_compiled",
        "smoke_tested",
        "database_provisioned",
        "database_migrated",
    ],
)
def test_existing_phases_are_unchanged(phase: str) -> None:
    payload = failure()
    payload.update(phase=phase, diagnostic_ref=f"{phase}:attempt-2")
    assert parse_build_failure(payload) == payload
