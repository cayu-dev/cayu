from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest
from examples.aws.lambda_microvm_agent.metadata_isolation_task import (
    REQUIRED_EVIDENCE_VALUES,
)
from tests.egress_conformance import registration_for


def _load_nightly_verification() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "scripts" / "nightly_verification.py"
    spec = importlib.util.spec_from_file_location("nightly_verification", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


nightly = _load_nightly_verification()


def test_prompt_cache_compaction_live_check_is_anthropic_credential_gated() -> None:
    check = next(
        check for check in nightly.CHECKS if check.id == "advanced-prompt-cache-compaction"
    )

    assert check.required_env == ("ANTHROPIC_API_KEY",)
    assert check.requires_provider_api_key is True
    assert "examples.prompt_cache_compaction.app" in check.command
    assert check.command[-4:] == ("--provider", "anthropic", "--trials", "1")


def test_missing_required_env_skips_without_running_command() -> None:
    called = False
    check = nightly.VerificationCheck(
        id="live",
        capability="live provider",
        lane="provider",
        command=("python", "example.py"),
        prerequisites=("TOKEN",),
        required_env=("TOKEN",),
    )

    def runner(command, env):
        nonlocal called
        called = True
        return nightly.CommandOutcome(returncode=0)

    result = nightly.run_checks([check], environ={}, runner=runner)[0]

    assert called is False
    assert result.status == nightly.STATUS_SKIPPED
    assert result.reason == "TOKEN is not set"


def test_any_env_requirement_accepts_either_key() -> None:
    check = nightly.VerificationCheck(
        id="provider-contract",
        capability="provider contract",
        lane="provider",
        command=("python", "example.py"),
        required_any_env=(("OPENAI_API_KEY", "ANTHROPIC_API_KEY"),),
    )

    result = nightly.run_checks(
        [check],
        environ={"ANTHROPIC_API_KEY": "set"},
        runner=lambda command, env: nightly.CommandOutcome(returncode=0, stdout="ok"),
    )[0]

    assert result.status == nightly.STATUS_VERIFIED


def test_provider_api_key_requirement_matches_selected_provider() -> None:
    called = False
    check = nightly.VerificationCheck(
        id="provider-contract",
        capability="provider contract",
        lane="provider",
        command=("python", "example.py"),
        required_any_env=(("OPENAI_API_KEY", "ANTHROPIC_API_KEY"),),
        requires_provider_api_key=True,
    )

    def runner(command, env):
        nonlocal called
        called = True
        return nightly.CommandOutcome(returncode=0)

    result = nightly.run_checks(
        [check],
        environ={"CAYU_PROVIDER": "openai", "ANTHROPIC_API_KEY": "set"},
        runner=runner,
    )[0]

    assert called is False
    assert result.status == nightly.STATUS_SKIPPED
    assert result.reason == "OPENAI_API_KEY is not set for CAYU_PROVIDER=openai"


def test_provider_api_key_requirement_accepts_matching_selected_provider() -> None:
    called = False
    check = nightly.VerificationCheck(
        id="provider-contract",
        capability="provider contract",
        lane="provider",
        command=("python", "example.py"),
        required_any_env=(("OPENAI_API_KEY", "ANTHROPIC_API_KEY"),),
        requires_provider_api_key=True,
    )

    def runner(command, env):
        nonlocal called
        called = True
        return nightly.CommandOutcome(returncode=0)

    result = nightly.run_checks(
        [check],
        environ={"CAYU_PROVIDER": "anthropic", "ANTHROPIC_API_KEY": "set"},
        runner=runner,
    )[0]

    assert called is True
    assert result.status == nightly.STATUS_VERIFIED


def test_successful_pytest_check_records_counts_and_env_overrides() -> None:
    check = nightly.VerificationCheck(
        id="postgres",
        capability="postgres",
        lane="postgres",
        command=("uv", "run", "pytest"),
        env={"CAYU_REQUIRE_POSTGRES": "1"},
    )
    observed_env = {}

    def runner(command, env):
        observed_env.update(env)
        return nightly.CommandOutcome(
            returncode=0,
            stdout="3 passed, 2 skipped in 0.12s",
        )

    result = nightly.run_checks([check], environ={}, runner=runner)[0]

    assert observed_env["CAYU_REQUIRE_POSTGRES"] == "1"
    assert result.status == nightly.STATUS_VERIFIED
    assert result.evidence == {"returncode": 0, "passed": 3, "skipped": 2}
    assert result.as_json()["command"] == "uv run pytest"


def test_run_checks_reports_progress() -> None:
    check = nightly.VerificationCheck(
        id="baseline",
        capability="baseline",
        lane="python",
        command=("uv", "run", "pytest"),
    )
    messages: list[str] = []

    result = nightly.run_checks(
        [check],
        environ={},
        runner=lambda command, env: nightly.CommandOutcome(returncode=0),
        progress=messages.append,
    )[0]

    assert result.status == nightly.STATUS_VERIFIED
    assert messages[0] == "[1/1] baseline (python) starting"
    assert messages[1].startswith("[1/1] baseline verified in ")


def test_progress_reason_prefers_exception_line() -> None:
    result = nightly.VerificationResult(
        capability="provider",
        check_id="live",
        lane="provider",
        status=nightly.STATUS_FAILED,
        command=("python", "example.py"),
        prerequisites=(),
        reason="           ^^^^^^^^^^^^^^^^\nE           RuntimeError: failed clearly",
    )

    assert nightly._progress_reason(result) == " (RuntimeError: failed clearly)"

    timeout_result = nightly.VerificationResult(
        capability="provider",
        check_id="live",
        lane="provider",
        status=nightly.STATUS_FAILED,
        command=("python", "example.py"),
        prerequisites=(),
        reason="stderr:\ncommand timed out after 0.01s",
    )

    assert nightly._progress_reason(timeout_result) == " (command timed out after 0.01s)"


def test_check_can_unset_inherited_live_credentials() -> None:
    check = nightly.VerificationCheck(
        id="baseline",
        capability="baseline",
        lane="python",
        command=("uv", "run", "pytest"),
        unset_env=("GEMINI_API_KEY",),
    )
    observed_env = {}

    def runner(command, env):
        observed_env.update(env)
        return nightly.CommandOutcome(returncode=0)

    result = nightly.run_checks(
        [check],
        environ={"GEMINI_API_KEY": "live-key", "PATH": "/bin"},
        runner=runner,
    )[0]

    assert result.status == nightly.STATUS_VERIFIED
    assert "GEMINI_API_KEY" not in observed_env
    assert observed_env["PATH"] == "/bin"


def test_failed_check_keeps_returncode_and_tail_reason() -> None:
    check = nightly.VerificationCheck(
        id="failing",
        capability="failure",
        lane="python",
        command=("uv", "run", "pytest"),
    )
    stderr = "\n".join(f"line {number}" for number in range(15))

    result = nightly.run_checks(
        [check],
        environ={},
        runner=lambda command, env: nightly.CommandOutcome(returncode=1, stderr=stderr),
    )[0]

    assert result.status == nightly.STATUS_FAILED
    assert result.evidence["returncode"] == 1
    assert result.reason == "stderr:\n" + "\n".join(f"line {number}" for number in range(3, 15))


def test_failed_check_includes_stdout_and_stderr_tails() -> None:
    check = nightly.VerificationCheck(
        id="failing",
        capability="failure",
        lane="python",
        command=("uv", "run", "pytest"),
    )

    result = nightly.run_checks(
        [check],
        environ={},
        runner=lambda command, env: nightly.CommandOutcome(
            returncode=1,
            stdout="assertion summary",
            stderr="warning before failure",
        ),
    )[0]

    assert result.status == nightly.STATUS_FAILED
    assert result.reason == "stderr:\nwarning before failure\nstdout:\nassertion summary"


def test_default_subprocess_runner_reports_timeout() -> None:
    check = nightly.VerificationCheck(
        id="slow",
        capability="slow command",
        lane="python",
        command=(sys.executable, "-c", "import time; time.sleep(10)"),
        timeout_s=0.01,
    )

    result = nightly.run_checks([check], environ=os.environ)[0]

    assert result.status == nightly.STATUS_FAILED
    assert result.evidence == {"returncode": 124, "timed_out": True}
    assert result.reason is not None
    assert "command timed out after 0.01s" in result.reason


def test_unclaimed_check_is_reported_without_command() -> None:
    check = nightly.VerificationCheck(
        id="gap",
        capability="uncovered capability",
        lane="gap",
        status_on_success=nightly.STATUS_UNCLAIMED,
        reason="not implemented",
    )

    result = nightly.run_checks([check], environ={}, runner=lambda command, env: None)[0]
    markdown = nightly.render_markdown([result])

    assert result.status == nightly.STATUS_UNCLAIMED
    assert result.reason == "not implemented"
    assert "| gap | gap | unclaimed | uncovered capability | not implemented |" in markdown


def test_markdown_report_escapes_multiline_cells() -> None:
    result = nightly.VerificationResult(
        capability="capability | with pipe",
        check_id="failing",
        lane="python",
        status=nightly.STATUS_FAILED,
        command=("uv", "run", "pytest"),
        prerequisites=(),
        reason="line one\nline | two",
    )

    markdown = nightly.render_markdown([result])

    assert "capability \\| with pipe" in markdown
    assert "line one<br>line \\| two" in markdown


def test_strict_policy_fails_on_unaccepted_statuses() -> None:
    results = [
        nightly.VerificationResult(
            capability=status,
            check_id=status,
            lane="lane",
            status=status,
            command=(),
            prerequisites=(),
        )
        for status in (
            nightly.STATUS_FAILED,
            nightly.STATUS_SKIPPED,
            nightly.STATUS_UNCLAIMED,
        )
    ]

    assert nightly._strict_failed(results) is True


def test_strict_policy_accepts_verified_statuses() -> None:
    results = [
        nightly.VerificationResult(
            capability=status,
            check_id=status,
            lane="lane",
            status=status,
            command=(),
            prerequisites=(),
        )
        for status in (nightly.STATUS_VERIFIED, nightly.STATUS_HERMETIC)
    ]

    assert nightly._strict_failed(results) is False


def test_check_rejects_unknown_success_status() -> None:
    with pytest.raises(ValueError, match="status_on_success"):
        nightly.VerificationCheck(
            id="invalid-status",
            capability="invalid status",
            lane="test",
            status_on_success="smoke",
        )


def test_strict_policy_rejects_unknown_result_status() -> None:
    result = nightly.VerificationResult(
        capability="unknown",
        check_id="unknown",
        lane="test",
        status="smoke",
        command=(),
        prerequisites=(),
    )

    assert nightly._strict_failed([result]) is True


def test_internal_evals_hermetic_check_pins_command_and_unsets_live_credentials() -> None:
    check = next(check for check in nightly.CHECKS if check.id == "internal-evals-hermetic")

    assert check.lane == "python"
    assert check.command == (
        "uv",
        "run",
        "cayu",
        "eval",
        "run",
        "cayu.evals.internal.runtime_acceptance:build",
        "--case-timeout-seconds",
        "30",
        "--output",
        ".cayu-internal-runtime-acceptance.json",
    )
    assert check.status_on_success == nightly.STATUS_HERMETIC
    assert check.unset_env == nightly._LIVE_CREDENTIAL_ENV


def test_console_pty_check_pins_standalone_nightly_command() -> None:
    check = next(check for check in nightly.CHECKS if check.id == "console-pty")

    assert check.lane == "cli"
    assert check.command == (
        "uv",
        "run",
        "--extra",
        "console",
        "--group",
        "nightly",
        "python",
        "scripts/console_pty_verification.py",
    )
    assert check.status_on_success == nightly.STATUS_HERMETIC
    assert check.prerequisites == ("POSIX PTY", "Cayu console extra")
    assert check.required_modules == ("pty",)
    assert check.requires_structured_evidence is True


def test_live_credential_policy_contains_aws_inputs() -> None:
    expected = {
        "AWS_ACCESS_KEY_ID",
        "AWS_DEFAULT_REGION",
        "AWS_PROFILE",
        "AWS_REGION",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "CAYU_BEDROCK_LIVE",
        "CAYU_BEDROCK_MODEL",
        "CAYU_LAMBDA_MICROVM_IMAGE",
        "CAYU_LAMBDA_MICROVM_LIVE",
    }

    assert expected <= set(nightly._LIVE_CREDENTIAL_ENV)


def test_bedrock_live_check_defers_credential_discovery_to_boto3(tmp_path: Path) -> None:
    check = next(check for check in nightly.CHECKS if check.id == "bedrock-provider-live")
    environ = {
        "HOME": str(tmp_path),
        "CAYU_BEDROCK_LIVE": "1",
        "CAYU_BEDROCK_MODEL": "anthropic.claude-test",
        "AWS_REGION": "us-west-2",
    }

    missing = nightly._missing_prerequisites(check, environ)

    assert missing == []


def test_bedrock_live_check_requires_explicit_enabled_value() -> None:
    check = next(check for check in nightly.CHECKS if check.id == "bedrock-provider-live")
    environ = {
        "CAYU_BEDROCK_LIVE": "0",
        "CAYU_BEDROCK_MODEL": "anthropic.claude-test",
        "AWS_REGION": "us-west-2",
    }

    assert nightly._missing_prerequisites(check, environ) == ["CAYU_BEDROCK_LIVE must equal '1'"]


@pytest.mark.parametrize(
    ("check_id", "lane", "test_path", "opt_in_env", "required_env"),
    [
        (
            "microsandbox-live-virtual-egress",
            "microsandbox",
            "tests/egress/test_microsandbox_egress_e2e.py",
            "CAYU_RUN_MICROSANDBOX_EGRESS_E2E",
            (),
        ),
        (
            "e2b-live-virtual-egress",
            "e2b",
            "tests/egress/test_e2b_egress_e2e.py",
            "CAYU_RUN_E2B_EGRESS_E2E",
            (
                "E2B_API_KEY",
                "CAYU_E2B_PROXY_EXPOSURE_COMMAND",
                "CAYU_E2B_PROXY_URL",
            ),
        ),
    ],
)
def test_virtual_egress_live_checks_are_registered_and_explicitly_gated(
    check_id: str,
    lane: str,
    test_path: str,
    opt_in_env: str,
    required_env: tuple[str, ...],
) -> None:
    check = next(check for check in nightly.CHECKS if check.id == check_id)
    registration = registration_for(lane)

    assert check.capability.endswith("virtual-egress enforcement")
    assert check.lane == lane
    assert check.command[:2] == ("uv", "run")
    assert test_path in check.command
    assert "-s" in check.command
    assert check.command[-2:] == ("-k", "shared_real_boundary_security_contract")
    assert registration.live_proof_source == test_path
    assert check.status_on_success == nightly.STATUS_VERIFIED
    assert check.requires_structured_evidence is True
    assert check.required_env == required_env
    assert check.required_env_values == {opt_in_env: "1"}
    assert check.required_modules == (lane,)

    result = nightly.run_checks(
        [check],
        environ={},
        runner=lambda command, env: pytest.fail("gated check unexpectedly ran"),
    )[0]

    assert result.status == nightly.STATUS_SKIPPED
    assert result.reason is not None
    assert f"{opt_in_env} is not set" in result.reason
    assert result.evidence["harness"] == {
        "schema": "cayu.egress_conformance.v1",
        "records": [
            {
                "adapter": lane,
                "scenario": "live-security-conformance",
                "status": "skipped",
                "proof_source": "nightly",
                "observations": [],
                "cleanup_outcome": "not-applicable",
                "duration_ms": 0,
                "reason": "prerequisites-unavailable",
            }
        ],
    }


def test_docker_virtual_egress_is_registered_as_uncredentialed_live_verification() -> None:
    check = next(check for check in nightly.CHECKS if check.id == "docker-live-virtual-egress")
    registration = registration_for("docker")

    assert check.lane == registration.runner_kind
    assert registration.live_proof_source in check.command
    assert check.required_env == ()
    assert check.required_env_values == {}
    assert check.requires_docker is True
    assert check.env == {"CAYU_REQUIRE_DOCKER_EGRESS": "1"}
    assert check.status_on_success == nightly.STATUS_VERIFIED
    assert "-s" in check.command
    assert check.command[-2:] == ("-k", "shared_real_boundary_security_contract")
    assert check.requires_structured_evidence is True


def test_microsandbox_guest_agent_liveness_is_registered_and_explicitly_gated() -> None:
    check = next(
        check for check in nightly.CHECKS if check.id == "microsandbox-live-guest-agent-liveness"
    )

    assert check.lane == "microsandbox"
    assert check.command == (
        "uv",
        "run",
        "python",
        "examples/microsandbox_guest_agent_liveness_live.py",
    )
    assert check.required_modules == ("microsandbox",)
    assert check.required_env_values == {"CAYU_RUN_MICROSANDBOX_GUEST_AGENT_LIVE": "1"}


def test_microsandbox_network_default_is_registered_and_explicitly_gated() -> None:
    check = next(
        check for check in nightly.CHECKS if check.id == "microsandbox-live-network-default"
    )

    assert check.lane == "microsandbox"
    assert check.command == (
        "uv",
        "run",
        "--group",
        "nightly",
        "--extra",
        "microsandbox",
        "python",
        "examples/microsandbox_network_default_live.py",
    )
    assert check.required_modules == ("microsandbox",)
    assert check.required_env_values == {"CAYU_RUN_MICROSANDBOX_NETWORK_LIVE": "1"}


def test_e2b_hardened_handoff_is_registered_and_explicitly_gated() -> None:
    check = next(check for check in nightly.CHECKS if check.id == "e2b-live-hardened-handoff")

    assert check.lane == "e2b"
    assert check.command == (
        "uv",
        "run",
        "--extra",
        "e2b",
        "--extra",
        "dev",
        "python",
        "-m",
        "pytest",
        "tests/runners/test_e2b_handoff_e2e.py",
        "-q",
        "-s",
    )
    assert check.status_on_success == nightly.STATUS_VERIFIED
    assert check.required_modules == ("e2b",)
    assert check.required_env == ("E2B_API_KEY",)
    assert check.required_env_values == {"CAYU_RUN_E2B_HANDOFF_E2E": "1"}


def test_virtual_egress_opt_in_flag_must_equal_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    check = next(check for check in nightly.CHECKS if check.id == "e2b-live-virtual-egress")
    called = False
    monkeypatch.setattr(nightly, "_module_missing", lambda name: False)

    def runner(command, env):
        nonlocal called
        called = True
        return nightly.CommandOutcome(returncode=0)

    result = nightly.run_checks(
        [check],
        environ={
            "CAYU_RUN_E2B_EGRESS_E2E": "0",
            "E2B_API_KEY": "set",
            "CAYU_E2B_PROXY_EXPOSURE_COMMAND": "tunnel {host} {port}",
            "CAYU_E2B_PROXY_URL": "http://proxy.example:8443",
        },
        runner=runner,
    )[0]

    assert called is False
    assert result.status == nightly.STATUS_SKIPPED
    assert result.reason == "CAYU_RUN_E2B_EGRESS_E2E must equal '1'"


def test_microsandbox_virtual_egress_runs_when_module_is_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    check = next(
        check for check in nightly.CHECKS if check.id == "microsandbox-live-virtual-egress"
    )
    called = False
    monkeypatch.setattr(nightly, "_module_missing", lambda name: False)

    def runner(command, env):
        nonlocal called
        called = True
        return nightly.CommandOutcome(
            returncode=0,
            stdout=(
                'CAYU_NIGHTLY_EVIDENCE={"schema":"cayu.egress_conformance.v1",'
                '"records":[{"adapter":"microsandbox","scenario":"guest-network-'
                'bypass-denial","status":"verified","proof_source":"live",'
                '"observations":["public-ip-denied","metadata-service-denied"],'
                '"cleanup_outcome":"complete","duration_ms":1,'
                '"reason":"contract-satisfied"}]}\n'
            ),
        )

    result = nightly.run_checks(
        [check],
        environ={"CAYU_RUN_MICROSANDBOX_EGRESS_E2E": "1"},
        runner=runner,
    )[0]

    assert called is True
    assert result.status == nightly.STATUS_VERIFIED
    assert result.evidence["harness"]["schema"] == "cayu.egress_conformance.v1"


def test_virtual_egress_failed_subprocess_preserves_emitted_failure_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    check = next(
        check for check in nightly.CHECKS if check.id == "microsandbox-live-virtual-egress"
    )
    monkeypatch.setattr(nightly, "_module_missing", lambda name: False)
    emitted = {
        "schema": "cayu.egress_conformance.v1",
        "records": [
            {
                "adapter": "microsandbox",
                "scenario": "live-security-conformance",
                "status": "failed",
                "proof_source": "live",
                "observations": [],
                "cleanup_outcome": "unknown",
                "duration_ms": 1,
                "reason": "check-failed",
            }
        ],
    }
    stdout = (
        "CAYU_NIGHTLY_EVIDENCE="
        '{"schema":"cayu.egress_conformance.v1","records":[{"adapter":'
        '"microsandbox","scenario":"live-security-conformance","status":"failed",'
        '"proof_source":"live","observations":[],"cleanup_outcome":"unknown",'
        '"duration_ms":1,"reason":"check-failed"}]}\n'
    )

    result = nightly.run_checks(
        [check],
        environ={"CAYU_RUN_MICROSANDBOX_EGRESS_E2E": "1"},
        runner=lambda command, env: nightly.CommandOutcome(returncode=1, stdout=stdout),
    )[0]

    assert result.status == nightly.STATUS_FAILED
    assert result.evidence["harness"] == emitted


def test_virtual_egress_failed_subprocess_without_emission_uses_nightly_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    check = next(
        check for check in nightly.CHECKS if check.id == "microsandbox-live-virtual-egress"
    )
    monkeypatch.setattr(nightly, "_module_missing", lambda name: False)

    result = nightly.run_checks(
        [check],
        environ={"CAYU_RUN_MICROSANDBOX_EGRESS_E2E": "1"},
        runner=lambda command, env: nightly.CommandOutcome(returncode=1),
    )[0]

    assert result.status == nightly.STATUS_FAILED
    assert result.evidence["harness"]["records"] == [
        {
            "adapter": "microsandbox",
            "scenario": "live-security-conformance",
            "status": "failed",
            "proof_source": "nightly",
            "observations": [],
            "cleanup_outcome": "unknown",
            "duration_ms": 0,
            "reason": "check-failed",
        }
    ]


def test_lambda_microvm_live_check_defers_credential_discovery_to_boto3(
    tmp_path: Path,
) -> None:
    check = next(check for check in nightly.CHECKS if check.id == "lambda-microvm-live")
    environ = {
        "HOME": str(tmp_path),
        "CAYU_LAMBDA_MICROVM_LIVE": "1",
        "CAYU_LAMBDA_MICROVM_IMAGE": "arn:aws:lambda:us-west-2:123:microvm-image:cayu",
        "AWS_REGION": "us-west-2",
    }

    missing = nightly._missing_prerequisites(check, environ)

    assert missing == []


def test_lambda_microvm_live_opt_in_flag_must_equal_one() -> None:
    check = next(check for check in nightly.CHECKS if check.id == "lambda-microvm-live")
    environ = {
        "CAYU_LAMBDA_MICROVM_LIVE": "0",
        "CAYU_LAMBDA_MICROVM_IMAGE": "arn:aws:lambda:us-west-2:123:microvm-image:cayu",
        "AWS_REGION": "us-west-2",
    }

    result = nightly.run_checks(
        [check],
        environ=environ,
        runner=lambda command, env: pytest.fail(
            "Lambda MicroVM live check ran without explicit opt-in"
        ),
    )[0]

    assert result.status == nightly.STATUS_SKIPPED
    assert result.reason == "CAYU_LAMBDA_MICROVM_LIVE must equal '1'"


def test_lambda_metadata_isolation_live_check_retains_verified_adapter_evidence() -> None:
    check = next(
        check
        for check in nightly.CHECKS
        if check.id == "aws-lambda-microvm-metadata-isolation-live"
    )
    evidence = {
        **REQUIRED_EVIDENCE_VALUES,
        "run_id": "metadata-isolation-nightly-wrapper",
        "credential_paths_checked": 7,
        "filesystem_files_inspected": 3,
        "processes_inspected": 2,
    }

    result = nightly.run_checks(
        [check],
        environ={
            "CAYU_AWS_METADATA_ISOLATION_LIVE": "1",
            "CAYU_AWS_METADATA_ISOLATION_STACK": "cayu-aws-agent",
            "AWS_REGION": "us-east-1",
        },
        runner=lambda command, env: nightly.CommandOutcome(
            returncode=0,
            stdout="CAYU_NIGHTLY_EVIDENCE=" + json.dumps(evidence),
        ),
    )[0]

    assert check.status_on_success == nightly.STATUS_VERIFIED
    assert "required metadata-isolation" in check.capability
    assert result.status == nightly.STATUS_VERIFIED
    assert result.evidence["harness"] == evidence


@pytest.mark.parametrize(
    ("check_id", "lane", "entrypoint"),
    [
        ("context-counting-live", "provider-contract", "examples/context_counting_live.py"),
        ("artifact-file-live", "provider-contract", "examples/artifact_file_live.py"),
        ("structured-output-live", "provider-contract", "examples/structured_output_live.py"),
        (
            "knowledge-embedding-live",
            "provider-embedding",
            "examples/knowledge_embedding_live.py",
        ),
        ("real-spend-budgets", "provider-spend", "examples/real_spend_budget_live.py"),
        ("dashboard-behavior", "dashboard", "examples/dashboard_behavior_live.py"),
        ("provider-stream-abort", "fault-injection", "tests/faults/test_provider_stream_abort.py"),
        ("sqlite-write-failure", "fault-injection", "tests/faults/test_sqlite_write_failure.py"),
        (
            "runner-cleanup-failure",
            "fault-injection",
            "tests/faults/test_runner_cleanup_failure.py",
        ),
        (
            "workspace-sync-failure",
            "fault-injection",
            "tests/faults/test_workspace_sync_failure.py",
        ),
    ],
)
def test_delivery_checks_are_verified_at_their_execution_boundary(
    check_id: str,
    lane: str,
    entrypoint: str,
) -> None:
    check = next(check for check in nightly.CHECKS if check.id == check_id)

    assert check.status_on_success == nightly.STATUS_VERIFIED
    assert check.lane == lane
    assert check.command[:2] == ("uv", "run")
    assert entrypoint in check.command
    assert check.reason is None
    if lane == "fault-injection":
        assert check.command[2] == "pytest"
        assert check.unset_env == nightly._LIVE_CREDENTIAL_ENV
        assert check.required_env == ()
        assert check.required_any_env == ()
    elif lane == "provider-contract":
        assert check.requires_provider_api_key is True
        assert check.required_any_env == (("OPENAI_API_KEY", "ANTHROPIC_API_KEY"),)
    elif lane in {"provider-embedding", "provider-spend"}:
        assert check.required_env == ("OPENAI_API_KEY",)
        assert check.requires_structured_evidence is True
    elif lane == "dashboard":
        assert check.unset_env == nightly._LIVE_CREDENTIAL_ENV
        assert check.requires_playwright_chromium is True
        assert check.requires_structured_evidence is True


@pytest.mark.parametrize(
    "check_id",
    [
        "context-pressure-calibration-live",
        "knowledge-recall-live",
        "knowledge-recall-many-live",
        "subagent-live",
        "subagent-parallel-live",
    ],
)
def test_demo_only_example_is_not_registered(check_id: str) -> None:
    assert check_id not in {check.id for check in nightly.CHECKS}


def test_smoke_status_constant_is_not_exposed() -> None:
    assert not hasattr(nightly, "STATUS_SMOKE")


def test_structured_harness_evidence_is_added_to_successful_result() -> None:
    check = nightly.VerificationCheck(
        id="structured-evidence",
        capability="structured evidence",
        lane="live",
        command=("live-check",),
        requires_structured_evidence=True,
    )
    stdout = 'progress\nCAYU_NIGHTLY_EVIDENCE={"maximum":"0.01","enforcement":"blocked"}\n'

    result = nightly.run_checks(
        [check],
        environ={},
        runner=lambda command, env: nightly.CommandOutcome(returncode=0, stdout=stdout),
    )[0]

    assert result.status == nightly.STATUS_VERIFIED
    assert result.evidence == {
        "returncode": 0,
        "harness": {"maximum": "0.01", "enforcement": "blocked"},
    }


@pytest.mark.parametrize(
    ("stdout", "reason"),
    [
        ("ordinary output\n", "required structured evidence was not emitted"),
        ("CAYU_NIGHTLY_EVIDENCE=not-json\n", "structured evidence is not valid JSON"),
        (
            "CAYU_NIGHTLY_EVIDENCE=[]\n",
            "structured evidence must be a JSON object",
        ),
        (
            "CAYU_NIGHTLY_EVIDENCE={}\nCAYU_NIGHTLY_EVIDENCE={}\n",
            "structured evidence was emitted more than once",
        ),
    ],
)
def test_required_structured_harness_evidence_fails_closed(
    stdout: str,
    reason: str,
) -> None:
    check = nightly.VerificationCheck(
        id="structured-evidence",
        capability="structured evidence",
        lane="live",
        command=("live-check",),
        requires_structured_evidence=True,
    )

    result = nightly.run_checks(
        [check],
        environ={},
        runner=lambda command, env: nightly.CommandOutcome(returncode=0, stdout=stdout),
    )[0]

    assert result.status == nightly.STATUS_FAILED
    assert result.reason == reason
    assert result.evidence == {"returncode": 0}


def test_dashboard_behavior_check_skips_when_playwright_chromium_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    check = next(check for check in nightly.CHECKS if check.id == "dashboard-behavior")
    called = False

    monkeypatch.setattr(nightly, "_module_missing", lambda name: False)
    monkeypatch.setattr(nightly, "_playwright_chromium_available", lambda: False)

    def runner(command, env):
        nonlocal called
        called = True
        return nightly.CommandOutcome(returncode=0)

    result = nightly.run_checks([check], environ={}, runner=runner)[0]

    assert called is False
    assert result.status == nightly.STATUS_SKIPPED
    assert result.reason == "Playwright Chromium is unavailable"


def test_internal_evals_hermetic_success_is_reported_without_live_credentials() -> None:
    check = next(check for check in nightly.CHECKS if check.id == "internal-evals-hermetic")
    environ = {
        "ANTHROPIC_API_KEY": "must-not-reach-command",
        "E2B_API_KEY": "must-not-reach-command",
        "GEMINI_API_KEY": "must-not-reach-command",
        "OPENAI_API_KEY": "must-not-reach-command",
    }

    def runner(command, effective_env):
        assert tuple(command) == check.command
        assert not set(environ).intersection(effective_env)
        return nightly.CommandOutcome(returncode=0)

    result = nightly.run_checks([check], environ=environ, runner=runner)[0]

    assert result == nightly.VerificationResult(
        capability=check.capability,
        check_id="internal-evals-hermetic",
        lane="python",
        status=nightly.STATUS_HERMETIC,
        command=check.command,
        prerequisites=(),
        evidence={"returncode": 0},
    )


def test_sigkill_recovery_check_pins_process_boundary_suite() -> None:
    check = next(check for check in nightly.CHECKS if check.id == "sigkill-recovery")

    assert check.lane == "recovery"
    assert check.command == (
        "uv",
        "run",
        "pytest",
        "tests/recovery/test_sigkill_recovery.py",
        "-q",
        "-m",
        "not postgres_recovery",
    )
    assert check.status_on_success == nightly.STATUS_VERIFIED
    assert check.prerequisites == ("POSIX SIGKILL",)
    assert check.requires_sigkill is True
    assert check.unset_env == nightly._LIVE_CREDENTIAL_ENV


def test_baseline_and_postgres_checks_partition_the_recovery_suite() -> None:
    core = next(check for check in nightly.CHECKS if check.id == "core-pytest")
    postgres = next(check for check in nightly.CHECKS if check.id == "postgres-required")

    assert core.command[-2:] == ("-m", "not sigkill_recovery")
    assert postgres.command[-2:] == (
        "-m",
        "not sigkill_recovery or postgres_recovery",
    )


def test_sigkill_recovery_skips_before_running_when_sigkill_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    check = next(check for check in nightly.CHECKS if check.id == "sigkill-recovery")
    called = False

    def runner(command, env):
        nonlocal called
        called = True
        return nightly.CommandOutcome(returncode=0)

    monkeypatch.setattr(nightly, "_sigkill_available", lambda: False)

    result = nightly.run_checks([check], environ={}, runner=runner)[0]

    assert called is False
    assert result.status == nightly.STATUS_SKIPPED
    assert result.reason == "POSIX SIGKILL is unavailable"


def test_strict_sigkill_selection_reports_command_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(nightly, "_sigkill_available", lambda: True)
    monkeypatch.setattr(
        nightly,
        "_run_subprocess",
        lambda command, env, timeout_s: nightly.CommandOutcome(
            returncode=1,
            stderr="SIGKILL recovery assertion failed",
        ),
    )

    exit_code = nightly.main(["--check", "sigkill-recovery", "--strict"])

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "| sigkill-recovery | recovery | failed |" in output
    assert "SIGKILL recovery assertion failed" in output


@pytest.mark.parametrize(
    ("check_id", "module"),
    [
        ("advanced-research-council", "cache_aware_research_council"),
        ("advanced-counterfactual-approval", "counterfactual_approval"),
        ("advanced-repo-tournament", "repo_maintainer_tournament"),
        ("advanced-tainted-incident", "tainted_incident_response"),
    ],
)
def test_advanced_examples_are_verified_gemini_contracts(
    check_id: str,
    module: str,
) -> None:
    check = next(check for check in nightly.CHECKS if check.id == check_id)

    assert check.lane == "advanced-runtime"
    assert check.command == (
        "uv",
        "run",
        "python",
        "-m",
        f"examples.{module}.app",
        "--mode",
        "live",
        "--provider",
        "gemini",
        "--trials",
        "5",
    )
    assert check.status_on_success == nightly.STATUS_VERIFIED
    assert check.required_env == ("GEMINI_API_KEY",)
    assert check.requires_provider_api_key is True
    assert check.requires_structured_evidence is True
    assert check.reason is None


@pytest.mark.parametrize(
    ("provider", "key_name"),
    [("openai", "OPENAI_API_KEY"), ("anthropic", "ANTHROPIC_API_KEY")],
)
@pytest.mark.parametrize(
    ("check_id", "module"),
    [
        ("advanced-research-council", "cache_aware_research_council"),
        ("advanced-counterfactual-approval", "counterfactual_approval"),
        ("advanced-repo-tournament", "repo_maintainer_tournament"),
        ("advanced-tainted-incident", "tainted_incident_response"),
    ],
)
def test_advanced_examples_have_credential_gated_provider_portability_checks(
    provider: str,
    key_name: str,
    check_id: str,
    module: str,
) -> None:
    check = next(check for check in nightly.CHECKS if check.id == f"{check_id}-{provider}")

    assert check.lane == "advanced-runtime-portability"
    assert check.command == (
        "uv",
        "run",
        "python",
        "-m",
        f"examples.{module}.app",
        "--mode",
        "live",
        "--provider",
        provider,
        "--trials",
        "1",
    )
    assert check.required_env == (key_name,)
    assert check.requires_provider_api_key is True
    assert check.requires_structured_evidence is True
