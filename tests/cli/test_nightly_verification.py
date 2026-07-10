from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest


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
        id="provider-smoke",
        capability="provider smoke",
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
        id="provider-smoke",
        capability="provider smoke",
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
        id="provider-smoke",
        capability="provider smoke",
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
            nightly.STATUS_SMOKE,
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
    assert set(check.unset_env) == {
        "ANTHROPIC_API_KEY",
        "E2B_API_KEY",
        "GEMINI_API_KEY",
        "OPENAI_API_KEY",
    }


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
    assert set(check.unset_env) == {
        "ANTHROPIC_API_KEY",
        "E2B_API_KEY",
        "GEMINI_API_KEY",
        "OPENAI_API_KEY",
    }


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
