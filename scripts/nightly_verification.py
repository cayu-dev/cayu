from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

STATUS_HERMETIC = "hermetic"
STATUS_VERIFIED = "verified"
STATUS_SKIPPED = "skipped"
STATUS_FAILED = "failed"
STATUS_UNCLAIMED = "unclaimed"

DEFAULT_CHECK_TIMEOUT_SECONDS = 1800.0
_TIMEOUT_RETURN_CODE = 124
_COUNT_RE = re.compile(r"(?P<count>\d+)\s+(?P<kind>passed|failed|skipped|error|errors)")
_STRUCTURED_EVIDENCE_PREFIX = "CAYU_NIGHTLY_EVIDENCE="
_LIVE_CREDENTIAL_ENV = (
    "ANTHROPIC_API_KEY",
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
    "E2B_API_KEY",
    "GEMINI_API_KEY",
    "OPENAI_API_KEY",
)
_SUCCESS_STATUSES = frozenset({STATUS_HERMETIC, STATUS_VERIFIED})
_PATH_ENV = {
    "docker": "CAYU_DOCKER_PATH",
    "sbx": "CAYU_SBX_PATH",
}


@dataclass(frozen=True)
class VerificationCheck:
    id: str
    capability: str
    lane: str
    command: tuple[str, ...] = ()
    status_on_success: str = STATUS_VERIFIED
    prerequisites: tuple[str, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict)
    unset_env: tuple[str, ...] = ()
    required_env: tuple[str, ...] = ()
    required_env_values: Mapping[str, str] = field(default_factory=dict)
    required_any_env: tuple[tuple[str, ...], ...] = ()
    required_commands: tuple[str, ...] = ()
    required_modules: tuple[str, ...] = ()
    requires_provider_api_key: bool = False
    requires_docker: bool = False
    requires_postgres: bool = False
    requires_microsandbox_runtime: bool = False
    requires_sigkill: bool = False
    requires_playwright_chromium: bool = False
    requires_structured_evidence: bool = False
    timeout_s: float | None = DEFAULT_CHECK_TIMEOUT_SECONDS
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.status_on_success in _SUCCESS_STATUSES:
            return
        if self.status_on_success == STATUS_UNCLAIMED and not self.command:
            return
        raise ValueError(
            "status_on_success must be verified or hermetic for executable checks, "
            "or unclaimed for a check without a command; "
            f"got {self.status_on_success!r}"
        )


@dataclass(frozen=True)
class CommandOutcome:
    returncode: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False


@dataclass(frozen=True)
class VerificationResult:
    capability: str
    check_id: str
    lane: str
    status: str
    command: tuple[str, ...]
    prerequisites: tuple[str, ...]
    reason: str | None = None
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def as_json(self) -> dict[str, Any]:
        return {
            "capability": self.capability,
            "check_id": self.check_id,
            "lane": self.lane,
            "status": self.status,
            "command": _format_command(self.command) if self.command else None,
            "prerequisites": list(self.prerequisites),
            "reason": self.reason,
            "evidence": dict(self.evidence),
        }


CommandRunner = Callable[[Sequence[str], Mapping[str, str]], CommandOutcome]
ProgressReporter = Callable[[str], None]

_ADVANCED_EXAMPLE_PORTABILITY_SPECS = (
    (
        "advanced-research-council",
        "cache_aware_research_council",
        "checkpoint forks, evaluator repair, causal budget, and cache-window policy",
    ),
    (
        "advanced-counterfactual-approval",
        "counterfactual_approval",
        "authority-free approval futures and exactly-once protected execution",
    ),
    (
        "advanced-repo-tournament",
        "repo_maintainer_tournament",
        "isolated repair tournament, evaluator gates, and idempotent PR promotion",
    ),
    (
        "advanced-tainted-incident",
        "tainted_incident_response",
        "fork-taint propagation, quarantine policy, and sanitized handoff",
    ),
)
_ADVANCED_PORTABILITY_PROVIDERS = (
    ("openai", "OPENAI_API_KEY"),
    ("anthropic", "ANTHROPIC_API_KEY"),
)


def _advanced_provider_portability_checks() -> tuple[VerificationCheck, ...]:
    return tuple(
        VerificationCheck(
            id=f"{check_id}-{provider}",
            capability=f"{capability} ({provider} portability)",
            lane="advanced-runtime-portability",
            command=(
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
            ),
            prerequisites=(key_name,),
            required_env=(key_name,),
            requires_provider_api_key=True,
            requires_structured_evidence=True,
        )
        for provider, key_name in _ADVANCED_PORTABILITY_PROVIDERS
        for check_id, module, capability in _ADVANCED_EXAMPLE_PORTABILITY_SPECS
    )


CHECKS: tuple[VerificationCheck, ...] = (
    VerificationCheck(
        id="core-pytest",
        capability="core runtime, stores, evals, server, local runner",
        lane="python",
        command=(
            "uv",
            "run",
            "pytest",
            "-q",
            "-rs",
            "-m",
            "not sigkill_recovery",
        ),
        status_on_success=STATUS_VERIFIED,
        unset_env=_LIVE_CREDENTIAL_ENV,
    ),
    VerificationCheck(
        id="internal-evals-hermetic",
        capability="first-party hermetic runtime acceptance evals",
        lane="python",
        command=(
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
        ),
        status_on_success=STATUS_HERMETIC,
        unset_env=_LIVE_CREDENTIAL_ENV,
    ),
    VerificationCheck(
        id="console-pty",
        capability="real Cayu console discovery, IPython async cells, and clean exit",
        lane="cli",
        command=(
            "uv",
            "run",
            "--extra",
            "console",
            "--group",
            "nightly",
            "python",
            "scripts/console_pty_verification.py",
        ),
        status_on_success=STATUS_HERMETIC,
        prerequisites=("POSIX PTY", "Cayu console extra"),
        unset_env=_LIVE_CREDENTIAL_ENV,
        required_modules=("pty",),
        requires_structured_evidence=True,
        timeout_s=60.0,
    ),
    VerificationCheck(
        id="postgres-required",
        capability="Postgres stores, migrations, pgvector, and dispatch claims",
        lane="postgres",
        command=(
            "uv",
            "run",
            "pytest",
            "-q",
            "-rs",
            "-m",
            "not sigkill_recovery or postgres_recovery",
        ),
        status_on_success=STATUS_VERIFIED,
        prerequisites=("Docker daemon or CAYU_TEST_POSTGRES_DSN",),
        env={"CAYU_REQUIRE_POSTGRES": "1"},
        unset_env=_LIVE_CREDENTIAL_ENV,
        requires_postgres=True,
    ),
    VerificationCheck(
        id="docker-runner",
        capability="DockerRunner real container exec and timeout cleanup",
        lane="docker",
        command=("uv", "run", "pytest", "tests/runners/test_docker_live.py", "-q"),
        status_on_success=STATUS_VERIFIED,
        prerequisites=("Docker daemon",),
        env={"CAYU_REQUIRE_DOCKER_RUNNER": "1"},
        requires_docker=True,
    ),
    VerificationCheck(
        id="docker-live-exec",
        capability="Docker live command interruption",
        lane="docker",
        command=("uv", "run", "python", "examples/docker_interrupt_live.py"),
        status_on_success=STATUS_VERIFIED,
        prerequisites=("Docker daemon",),
        requires_docker=True,
    ),
    VerificationCheck(
        id="docker-live-sync",
        capability="Docker SyncBinding round trip",
        lane="docker",
        command=("uv", "run", "python", "examples/docker_sync_binding_live.py"),
        status_on_success=STATUS_VERIFIED,
        prerequisites=("Docker daemon",),
        requires_docker=True,
    ),
    VerificationCheck(
        id="sbx-live-exec",
        capability="sbx live command interruption",
        lane="sbx",
        command=("uv", "run", "python", "examples/sbx_interrupt_live.py"),
        status_on_success=STATUS_VERIFIED,
        prerequisites=("sbx CLI",),
        required_commands=("sbx",),
    ),
    VerificationCheck(
        id="sbx-live-sync",
        capability="sbx SyncBinding round trip",
        lane="sbx",
        command=("uv", "run", "python", "examples/sbx_sync_binding_live.py"),
        status_on_success=STATUS_VERIFIED,
        prerequisites=("sbx CLI",),
        required_commands=("sbx",),
    ),
    VerificationCheck(
        id="microsandbox-live-runner",
        capability="MicrosandboxRunner real sandbox exec and cancellation cleanup",
        lane="microsandbox",
        command=("uv", "run", "python", "examples/microsandbox_runner_live.py"),
        status_on_success=STATUS_VERIFIED,
        prerequisites=("microsandbox package/runtime",),
        required_modules=("microsandbox",),
    ),
    VerificationCheck(
        id="microsandbox-live-runtime",
        capability="Microsandbox runtime environment",
        lane="microsandbox",
        command=("uv", "run", "python", "examples/microsandbox_runtime_live.py"),
        status_on_success=STATUS_VERIFIED,
        prerequisites=("microsandbox package/runtime",),
        required_modules=("microsandbox",),
    ),
    VerificationCheck(
        id="microsandbox-live-workspace",
        capability="Microsandbox workspace read/write",
        lane="microsandbox",
        command=("uv", "run", "python", "examples/microsandbox_workspace_live.py"),
        status_on_success=STATUS_VERIFIED,
        prerequisites=("microsandbox package/runtime",),
        required_modules=("microsandbox",),
    ),
    VerificationCheck(
        id="microsandbox-live-sync",
        capability="Microsandbox SyncBinding round trip",
        lane="microsandbox",
        command=("uv", "run", "python", "examples/microsandbox_sync_binding_live.py"),
        status_on_success=STATUS_VERIFIED,
        prerequisites=("microsandbox package/runtime",),
        required_modules=("microsandbox",),
    ),
    VerificationCheck(
        id="microsandbox-live-virtual-egress",
        capability="Microsandbox virtual-egress enforcement",
        lane="microsandbox",
        command=(
            "uv",
            "run",
            "--extra",
            "egress",
            "--extra",
            "microsandbox",
            "pytest",
            "tests/egress/test_microsandbox_egress_e2e.py",
            "-q",
        ),
        status_on_success=STATUS_VERIFIED,
        prerequisites=(
            "CAYU_RUN_MICROSANDBOX_EGRESS_E2E=1",
            "microsandbox package/runtime",
        ),
        required_env_values={"CAYU_RUN_MICROSANDBOX_EGRESS_E2E": "1"},
        required_modules=("microsandbox",),
        requires_microsandbox_runtime=True,
    ),
    VerificationCheck(
        id="e2b-live-runner",
        capability="E2BRunner real sandbox exec and cancellation cleanup",
        lane="e2b",
        command=("uv", "run", "python", "examples/e2b_runner_live.py"),
        status_on_success=STATUS_VERIFIED,
        prerequisites=("E2B_API_KEY", "e2b package"),
        required_env=("E2B_API_KEY",),
        required_modules=("e2b",),
    ),
    VerificationCheck(
        id="e2b-live-workspace",
        capability="E2B workspace read/write",
        lane="e2b",
        command=("uv", "run", "python", "examples/e2b_workspace_live.py"),
        status_on_success=STATUS_VERIFIED,
        prerequisites=("E2B_API_KEY", "e2b package"),
        required_env=("E2B_API_KEY",),
        required_modules=("e2b",),
    ),
    VerificationCheck(
        id="e2b-live-sync",
        capability="E2B SyncBinding round trip",
        lane="e2b",
        command=("uv", "run", "python", "examples/e2b_sync_binding_live.py"),
        status_on_success=STATUS_VERIFIED,
        prerequisites=("E2B_API_KEY", "e2b package"),
        required_env=("E2B_API_KEY",),
        required_modules=("e2b",),
    ),
    VerificationCheck(
        id="e2b-live-virtual-egress",
        capability="E2B virtual-egress enforcement",
        lane="e2b",
        command=(
            "uv",
            "run",
            "--extra",
            "egress",
            "--extra",
            "e2b",
            "pytest",
            "tests/egress/test_e2b_egress_e2e.py",
            "-q",
        ),
        status_on_success=STATUS_VERIFIED,
        prerequisites=(
            "CAYU_RUN_E2B_EGRESS_E2E=1",
            "E2B_API_KEY",
            "CAYU_E2B_PROXY_EXPOSURE_COMMAND",
            "CAYU_E2B_PROXY_URL with IPv4-literal host",
            "e2b package",
        ),
        required_env=(
            "E2B_API_KEY",
            "CAYU_E2B_PROXY_EXPOSURE_COMMAND",
            "CAYU_E2B_PROXY_URL",
        ),
        required_env_values={"CAYU_RUN_E2B_EGRESS_E2E": "1"},
        required_modules=("e2b",),
    ),
    VerificationCheck(
        id="lambda-microvm-live",
        capability="AWS Lambda MicroVM runner, workspace, cleanup, and suspend/resume",
        lane="aws-lambda-microvm",
        command=(
            "uv",
            "run",
            "--extra",
            "aws",
            "python",
            "examples/lambda_microvm_runner_live.py",
        ),
        status_on_success=STATUS_VERIFIED,
        prerequisites=(
            "CAYU_LAMBDA_MICROVM_LIVE=1",
            "CAYU_LAMBDA_MICROVM_IMAGE",
            "AWS_REGION or AWS_DEFAULT_REGION",
            "AWS credential chain",
        ),
        required_env=("CAYU_LAMBDA_MICROVM_IMAGE",),
        required_env_values={"CAYU_LAMBDA_MICROVM_LIVE": "1"},
        required_any_env=(("AWS_REGION", "AWS_DEFAULT_REGION"),),
        requires_structured_evidence=True,
    ),
    VerificationCheck(
        id="gemini-eval",
        capability="Chat Completions eval path against Gemini",
        lane="chat-completions",
        command=(
            "uv",
            "run",
            "pytest",
            "-q",
            "-rs",
            "tests/evals/test_runtime_evals.py::test_integration_eval_against_gemini",
        ),
        status_on_success=STATUS_VERIFIED,
        prerequisites=("GEMINI_API_KEY",),
        required_env=("GEMINI_API_KEY",),
    ),
    VerificationCheck(
        id="chat-completions-contract",
        capability="Chat Completions tool-call and structured-output contract",
        lane="chat-completions",
        command=("uv", "run", "python", "examples/chat_completions_contract_live.py"),
        status_on_success=STATUS_VERIFIED,
        prerequisites=("GEMINI_API_KEY",),
        required_env=("GEMINI_API_KEY",),
    ),
    VerificationCheck(
        id="structured-output-live",
        capability="OpenAI/Anthropic structured-output contract",
        lane="provider-contract",
        command=("uv", "run", "python", "examples/structured_output_live.py"),
        status_on_success=STATUS_VERIFIED,
        prerequisites=("OPENAI_API_KEY or ANTHROPIC_API_KEY",),
        required_any_env=(("OPENAI_API_KEY", "ANTHROPIC_API_KEY"),),
        requires_provider_api_key=True,
        requires_structured_evidence=True,
    ),
    VerificationCheck(
        id="bedrock-provider-live",
        capability="Amazon Bedrock text, tool structured output, usage, and token counting",
        lane="provider-contract",
        command=("uv", "run", "--extra", "aws", "python", "examples/bedrock_provider_live.py"),
        status_on_success=STATUS_VERIFIED,
        prerequisites=(
            "CAYU_BEDROCK_LIVE=1",
            "CAYU_BEDROCK_MODEL",
            "AWS_REGION or AWS_DEFAULT_REGION",
            "AWS credential chain",
        ),
        required_env=("CAYU_BEDROCK_MODEL",),
        required_env_values={"CAYU_BEDROCK_LIVE": "1"},
        required_any_env=(("AWS_REGION", "AWS_DEFAULT_REGION"),),
        requires_structured_evidence=True,
    ),
    VerificationCheck(
        id="artifact-file-live",
        capability="OpenAI/Anthropic artifact file contract",
        lane="provider-contract",
        command=("uv", "run", "python", "examples/artifact_file_live.py"),
        status_on_success=STATUS_VERIFIED,
        prerequisites=("OPENAI_API_KEY or ANTHROPIC_API_KEY", "Pillow and pypdf"),
        required_any_env=(("OPENAI_API_KEY", "ANTHROPIC_API_KEY"),),
        required_modules=("PIL", "pypdf"),
        requires_provider_api_key=True,
    ),
    VerificationCheck(
        id="context-counting-live",
        capability="OpenAI/Anthropic context counting contract",
        lane="provider-contract",
        command=("uv", "run", "python", "examples/context_counting_live.py"),
        status_on_success=STATUS_VERIFIED,
        prerequisites=("OPENAI_API_KEY or ANTHROPIC_API_KEY",),
        required_any_env=(("OPENAI_API_KEY", "ANTHROPIC_API_KEY"),),
        requires_provider_api_key=True,
    ),
    VerificationCheck(
        id="knowledge-embedding-live",
        capability="OpenAI embedding and semantic-retrieval contract",
        lane="provider-embedding",
        command=("uv", "run", "python", "examples/knowledge_embedding_live.py"),
        status_on_success=STATUS_VERIFIED,
        prerequisites=("OPENAI_API_KEY",),
        env={"CAYU_PROVIDER": "openai"},
        required_env=("OPENAI_API_KEY",),
        requires_structured_evidence=True,
    ),
    VerificationCheck(
        id="advanced-research-council",
        capability="checkpoint forks, evaluator repair, causal budget, and cache-window policy",
        lane="advanced-runtime",
        command=(
            "uv",
            "run",
            "python",
            "-m",
            "examples.cache_aware_research_council.app",
            "--mode",
            "live",
            "--provider",
            "gemini",
            "--trials",
            "5",
        ),
        status_on_success=STATUS_VERIFIED,
        prerequisites=("GEMINI_API_KEY",),
        required_env=("GEMINI_API_KEY",),
        requires_provider_api_key=True,
        requires_structured_evidence=True,
    ),
    VerificationCheck(
        id="advanced-counterfactual-approval",
        capability="authority-free approval futures and exactly-once protected execution",
        lane="advanced-runtime",
        command=(
            "uv",
            "run",
            "python",
            "-m",
            "examples.counterfactual_approval.app",
            "--mode",
            "live",
            "--provider",
            "gemini",
            "--trials",
            "5",
        ),
        status_on_success=STATUS_VERIFIED,
        prerequisites=("GEMINI_API_KEY",),
        required_env=("GEMINI_API_KEY",),
        requires_provider_api_key=True,
        requires_structured_evidence=True,
    ),
    VerificationCheck(
        id="advanced-repo-tournament",
        capability="isolated repair tournament, evaluator gates, and idempotent PR promotion",
        lane="advanced-runtime",
        command=(
            "uv",
            "run",
            "python",
            "-m",
            "examples.repo_maintainer_tournament.app",
            "--mode",
            "live",
            "--provider",
            "gemini",
            "--trials",
            "5",
        ),
        status_on_success=STATUS_VERIFIED,
        prerequisites=("GEMINI_API_KEY",),
        required_env=("GEMINI_API_KEY",),
        requires_provider_api_key=True,
        requires_structured_evidence=True,
    ),
    VerificationCheck(
        id="advanced-tainted-incident",
        capability="fork-taint propagation, quarantine policy, and sanitized handoff",
        lane="advanced-runtime",
        command=(
            "uv",
            "run",
            "python",
            "-m",
            "examples.tainted_incident_response.app",
            "--mode",
            "live",
            "--provider",
            "gemini",
            "--trials",
            "5",
        ),
        status_on_success=STATUS_VERIFIED,
        prerequisites=("GEMINI_API_KEY",),
        required_env=("GEMINI_API_KEY",),
        requires_provider_api_key=True,
        requires_structured_evidence=True,
    ),
    *_advanced_provider_portability_checks(),
    VerificationCheck(
        id="dashboard-behavior",
        capability="dashboard browser behavior",
        lane="dashboard",
        command=("uv", "run", "python", "examples/dashboard_behavior_live.py"),
        status_on_success=STATUS_VERIFIED,
        prerequisites=("Playwright Chromium", "Cayu server extra"),
        unset_env=_LIVE_CREDENTIAL_ENV,
        required_modules=("playwright", "fastapi", "uvicorn"),
        requires_playwright_chromium=True,
        requires_structured_evidence=True,
    ),
    VerificationCheck(
        id="sigkill-recovery",
        capability="crash recovery across a real process boundary",
        lane="recovery",
        command=(
            "uv",
            "run",
            "pytest",
            "tests/recovery/test_sigkill_recovery.py",
            "-q",
            "-m",
            "not postgres_recovery",
        ),
        status_on_success=STATUS_VERIFIED,
        prerequisites=("POSIX SIGKILL",),
        unset_env=_LIVE_CREDENTIAL_ENV,
        requires_sigkill=True,
    ),
    VerificationCheck(
        id="real-spend-budgets",
        capability="budgets under real provider spend",
        lane="provider-spend",
        command=("uv", "run", "python", "examples/real_spend_budget_live.py"),
        status_on_success=STATUS_VERIFIED,
        prerequisites=("OPENAI_API_KEY",),
        env={"CAYU_PROVIDER": "openai"},
        required_env=("OPENAI_API_KEY",),
        requires_structured_evidence=True,
    ),
    VerificationCheck(
        id="provider-stream-abort",
        capability="provider transport abort before terminal stream event",
        lane="fault-injection",
        command=(
            "uv",
            "run",
            "pytest",
            "tests/faults/test_provider_stream_abort.py",
            "-q",
        ),
        status_on_success=STATUS_VERIFIED,
        prerequisites=("loopback TCP", "SQLite"),
        unset_env=_LIVE_CREDENTIAL_ENV,
    ),
    VerificationCheck(
        id="sqlite-write-failure",
        capability="SQLite terminal-event write failure and manual recovery",
        lane="fault-injection",
        command=(
            "uv",
            "run",
            "pytest",
            "tests/faults/test_sqlite_write_failure.py",
            "-q",
        ),
        status_on_success=STATUS_VERIFIED,
        prerequisites=("SQLite triggers", "durable filesystem"),
        unset_env=_LIVE_CREDENTIAL_ENV,
    ),
    VerificationCheck(
        id="runner-cleanup-failure",
        capability="runner command cleanup failure with unknown subprocess state",
        lane="fault-injection",
        command=(
            "uv",
            "run",
            "pytest",
            "tests/faults/test_runner_cleanup_failure.py",
            "-q",
        ),
        status_on_success=STATUS_VERIFIED,
        prerequisites=("POSIX process groups",),
        unset_env=_LIVE_CREDENTIAL_ENV,
        requires_sigkill=True,
    ),
    VerificationCheck(
        id="workspace-sync-failure",
        capability="partial workspace sync finalization and convergent retry",
        lane="fault-injection",
        command=(
            "uv",
            "run",
            "pytest",
            "tests/faults/test_workspace_sync_failure.py",
            "-q",
        ),
        status_on_success=STATUS_VERIFIED,
        prerequisites=("SQLite", "durable filesystem"),
        unset_env=_LIVE_CREDENTIAL_ENV,
    ),
)


def run_checks(
    checks: Sequence[VerificationCheck],
    *,
    environ: Mapping[str, str] | None = None,
    runner: CommandRunner | None = None,
    progress: ProgressReporter | None = None,
) -> list[VerificationResult]:
    env = os.environ if environ is None else environ
    results: list[VerificationResult] = []
    total = len(checks)
    for index, check in enumerate(checks, start=1):
        if progress is not None:
            progress(f"[{index}/{total}] {check.id} ({check.lane}) starting")
        started = time.monotonic()
        result = _run_check(check, env, runner)
        elapsed_s = time.monotonic() - started
        if progress is not None:
            progress(
                f"[{index}/{total}] {check.id} {result.status} in {elapsed_s:.1f}s"
                f"{_progress_reason(result)}"
            )
        results.append(result)
    return results


def render_markdown(results: Sequence[VerificationResult]) -> str:
    counts: dict[str, int] = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1

    lines = [
        "# Nightly verification report",
        "",
        f"Generated: {datetime.now(UTC).isoformat()}",
        "",
        "## Summary",
        "",
        "| status | count |",
        "| --- | ---: |",
    ]
    for status in (
        STATUS_FAILED,
        STATUS_SKIPPED,
        STATUS_UNCLAIMED,
        STATUS_VERIFIED,
        STATUS_HERMETIC,
    ):
        if status in counts:
            lines.append(f"| {status} | {counts[status]} |")

    lines += [
        "",
        "## Checks",
        "",
        "| check | lane | status | capability | reason |",
        "| --- | --- | --- | --- | --- |",
    ]
    for result in results:
        reason = _markdown_cell(result.reason or "")
        lines.append(
            "| "
            f"{_markdown_cell(result.check_id)} | "
            f"{_markdown_cell(result.lane)} | "
            f"{_markdown_cell(result.status)} | "
            f"{_markdown_cell(result.capability)} | "
            f"{reason} |"
        )
    return "\n".join(lines) + "\n"


def report_payload(results: Sequence[VerificationResult]) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "results": [result.as_json() for result in results],
    }


def selected_checks(ids: Sequence[str]) -> list[VerificationCheck]:
    by_id = {check.id: check for check in CHECKS}
    if not ids:
        return list(CHECKS)

    unknown = sorted(set(ids) - set(by_id))
    if unknown:
        joined = ", ".join(unknown)
        raise SystemExit(f"Unknown check id(s): {joined}")
    return [by_id[id_] for id_ in ids]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Cayu nightly verification checks.")
    parser.add_argument("--check", action="append", default=[], help="Check id to run.")
    parser.add_argument("--list", action="store_true", help="List check ids and exit.")
    parser.add_argument("--json", type=Path, help="Write machine-readable JSON report.")
    parser.add_argument("--markdown", type=Path, help="Write Markdown report.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero on failed, skipped, or unclaimed checks.",
    )
    args = parser.parse_args(argv)

    if args.list:
        for check in CHECKS:
            print(f"{check.id}\t{check.lane}\t{check.capability}")
        return 0

    results = run_checks(selected_checks(args.check), progress=_log_progress)
    markdown = render_markdown(results)

    if args.json is not None:
        args.json.write_text(json.dumps(report_payload(results), indent=2) + "\n", encoding="utf-8")
    if args.markdown is not None:
        args.markdown.write_text(markdown, encoding="utf-8")
    if args.json is None and args.markdown is None:
        print(markdown, end="")

    if args.strict and _strict_failed(results):
        return 1
    return 0


def _run_check(
    check: VerificationCheck,
    environ: Mapping[str, str],
    runner: CommandRunner | None,
) -> VerificationResult:
    if not check.command:
        return VerificationResult(
            capability=check.capability,
            check_id=check.id,
            lane=check.lane,
            status=STATUS_UNCLAIMED,
            command=(),
            prerequisites=check.prerequisites,
            reason=check.reason,
        )

    effective_env = _effective_env(check, environ)
    missing = _missing_prerequisites(check, effective_env)
    if missing:
        return VerificationResult(
            capability=check.capability,
            check_id=check.id,
            lane=check.lane,
            status=STATUS_SKIPPED,
            command=check.command,
            prerequisites=check.prerequisites,
            reason="; ".join(missing),
        )

    if runner is None:
        outcome = _run_subprocess(check.command, effective_env, timeout_s=check.timeout_s)
    else:
        outcome = runner(check.command, effective_env)
    evidence = {
        "returncode": outcome.returncode,
        **_pytest_counts(outcome.stdout + "\n" + outcome.stderr),
    }
    if outcome.timed_out:
        evidence["timed_out"] = True
    if outcome.returncode == 0:
        if check.requires_structured_evidence:
            harness_evidence, evidence_error = _structured_evidence(outcome.stdout)
            if evidence_error is not None:
                return VerificationResult(
                    capability=check.capability,
                    check_id=check.id,
                    lane=check.lane,
                    status=STATUS_FAILED,
                    command=check.command,
                    prerequisites=check.prerequisites,
                    reason=evidence_error,
                    evidence=evidence,
                )
            evidence["harness"] = harness_evidence
        return VerificationResult(
            capability=check.capability,
            check_id=check.id,
            lane=check.lane,
            status=check.status_on_success,
            command=check.command,
            prerequisites=check.prerequisites,
            reason=check.reason,
            evidence=evidence,
        )
    return VerificationResult(
        capability=check.capability,
        check_id=check.id,
        lane=check.lane,
        status=STATUS_FAILED,
        command=check.command,
        prerequisites=check.prerequisites,
        reason=_failure_reason(outcome),
        evidence=evidence,
    )


def _effective_env(
    check: VerificationCheck,
    environ: Mapping[str, str],
) -> dict[str, str]:
    effective = {**environ, **check.env}
    for name in check.unset_env:
        effective.pop(name, None)
    return effective


def _missing_prerequisites(check: VerificationCheck, environ: Mapping[str, str]) -> list[str]:
    missing = [f"{name} is not set" for name in check.required_env if not environ.get(name)]
    for name, expected in check.required_env_values.items():
        actual = environ.get(name)
        if not actual:
            missing.append(f"{name} is not set")
        elif actual != expected:
            missing.append(f"{name} must equal {expected!r}")
    missing += [
        f"one of {', '.join(names)} must be set"
        for names in check.required_any_env
        if not any(environ.get(name) for name in names)
    ]
    missing_modules = [name for name in check.required_modules if _module_missing(name)]
    missing += [f"Python module {name!r} is unavailable" for name in missing_modules]
    missing += [
        f"command {name!r} is unavailable"
        for name in check.required_commands
        if _command_missing(name, environ)
    ]
    if check.requires_provider_api_key:
        missing += _provider_api_key_missing(environ)
    if check.requires_docker and not _docker_available(environ):
        missing.append("Docker daemon is unavailable")
    if check.requires_postgres and not _postgres_available(environ):
        missing.append("Postgres is unavailable: set CAYU_TEST_POSTGRES_DSN or run Docker")
    if (
        check.requires_microsandbox_runtime
        and "microsandbox" not in missing_modules
        and not _microsandbox_runtime_available()
    ):
        missing.append("Microsandbox runtime is unavailable")
    if check.requires_sigkill and not _sigkill_available():
        missing.append("POSIX SIGKILL is unavailable")
    if (
        check.requires_playwright_chromium
        and "playwright" not in missing_modules
        and not _playwright_chromium_available()
    ):
        missing.append("Playwright Chromium is unavailable")
    return missing


def _module_missing(name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(name) is None


def _microsandbox_runtime_available() -> bool:
    try:
        import microsandbox

        return bool(microsandbox.is_installed())
    except Exception:
        return False


def _playwright_chromium_available() -> bool:
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as playwright:
            return Path(playwright.chromium.executable_path).is_file()
    except Exception:
        return False


def _sigkill_available() -> bool:
    return os.name == "posix" and hasattr(signal, "SIGKILL")


def _command_missing(name: str, environ: Mapping[str, str]) -> bool:
    override = _PATH_ENV.get(name)
    if override and environ.get(override):
        return False
    return shutil.which(name) is None


def _provider_api_key_missing(environ: Mapping[str, str]) -> list[str]:
    provider = environ.get("CAYU_PROVIDER")
    if provider is not None:
        provider = provider.strip().lower()
    if provider == "openai":
        return (
            []
            if environ.get("OPENAI_API_KEY")
            else ["OPENAI_API_KEY is not set for CAYU_PROVIDER=openai"]
        )
    if provider == "anthropic":
        return (
            []
            if environ.get("ANTHROPIC_API_KEY")
            else ["ANTHROPIC_API_KEY is not set for CAYU_PROVIDER=anthropic"]
        )
    if provider:
        return ["CAYU_PROVIDER must be openai or anthropic"]
    return []


def _postgres_available(environ: Mapping[str, str]) -> bool:
    if environ.get("CAYU_TEST_POSTGRES_DSN"):
        return True
    return _docker_available(environ)


def _docker_available(environ: Mapping[str, str]) -> bool:
    docker_path = environ.get("CAYU_DOCKER_PATH") or shutil.which("docker")
    if docker_path is None:
        return False
    try:
        result = subprocess.run(
            [docker_path, "info"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def _run_subprocess(
    command: Sequence[str],
    env: Mapping[str, str],
    *,
    timeout_s: float | None = DEFAULT_CHECK_TIMEOUT_SECONDS,
) -> CommandOutcome:
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            env=dict(env),
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = _timeout_output(exc.stdout)
        stderr = _timeout_output(exc.stderr)
        timeout_line = f"command timed out after {timeout_s:g}s"
        stderr = f"{timeout_line}\n{stderr}".rstrip()
        return CommandOutcome(
            returncode=_TIMEOUT_RETURN_CODE,
            stdout=stdout,
            stderr=stderr,
            timed_out=True,
        )
    return CommandOutcome(
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _pytest_counts(output: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for match in _COUNT_RE.finditer(output):
        kind = match.group("kind")
        if kind == "errors":
            kind = "error"
        counts[kind] = counts.get(kind, 0) + int(match.group("count"))
    return counts


def _structured_evidence(output: str) -> tuple[dict[str, Any] | None, str | None]:
    encoded = [
        line.removeprefix(_STRUCTURED_EVIDENCE_PREFIX)
        for line in output.splitlines()
        if line.startswith(_STRUCTURED_EVIDENCE_PREFIX)
    ]
    if not encoded:
        return None, "required structured evidence was not emitted"
    if len(encoded) > 1:
        return None, "structured evidence was emitted more than once"
    try:
        evidence = json.loads(encoded[0])
    except json.JSONDecodeError:
        return None, "structured evidence is not valid JSON"
    if type(evidence) is not dict:
        return None, "structured evidence must be a JSON object"
    return evidence, None


def _failure_reason(outcome: CommandOutcome) -> str:
    parts: list[str] = []
    stderr = outcome.stderr.strip()
    stdout = outcome.stdout.strip()
    if stderr:
        parts.append(f"stderr:\n{_tail_lines(stderr)}")
    if stdout:
        parts.append(f"stdout:\n{_tail_lines(stdout)}")
    if not parts:
        return f"command exited with status {outcome.returncode}"
    return "\n".join(parts)


def _tail_lines(text: str, *, limit: int = 12) -> str:
    return "\n".join(text.splitlines()[-limit:])


def _timeout_output(output: str | bytes | None) -> str:
    if output is None:
        return ""
    if isinstance(output, bytes):
        return output.decode("utf-8", errors="replace")
    return output


def _log_progress(message: str) -> None:
    print(f"nightly: {message}", file=sys.stderr, flush=True)


def _progress_reason(result: VerificationResult) -> str:
    if result.status in {STATUS_FAILED, STATUS_SKIPPED, STATUS_UNCLAIMED} and result.reason:
        return f" ({_reason_summary(result.reason)})"
    return ""


def _reason_summary(reason: str) -> str:
    lines = [
        line.strip()
        for line in reason.splitlines()
        if line.strip() and line.strip() not in {"stderr:", "stdout:"}
    ]
    for line in lines:
        if line.startswith("E           "):
            return line.removeprefix("E           ")
        if "Error:" in line or line.endswith(" is not set"):
            return line
    return lines[0] if lines else ""


def _format_command(command: Sequence[str]) -> str:
    return shlex.join(command)


def _markdown_cell(value: str) -> str:
    return value.replace("|", r"\|").replace("\n", "<br>")


def _strict_failed(results: Sequence[VerificationResult]) -> bool:
    return any(result.status not in _SUCCESS_STATUSES for result in results)


if __name__ == "__main__":
    raise SystemExit(main())
