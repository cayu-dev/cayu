#!/usr/bin/env python3
"""Run Cayu's canonical CI lanes or orchestrate them locally with proof.

This helper never calls GitHub, fetches or pushes remotes, posts comments, or
mutates remote repository state. GitHub Actions invokes its individual lanes;
local verification invokes the same lane functions serially.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import re
import resource
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from collections import deque
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import NoReturn

_GENERAL_TEST_ENV = {
    "CAYU_REQUIRE_POSTGRES": "1",
    "CAYU_REQUIRE_DOCKER_RUNNER": "1",
    "CAYU_REQUIRE_DOCKER_EGRESS": "1",
    "CAYU_REQUIRE_CURRENT_TEST_DURATIONS": "1",
}
_SPECIALIST_TEST_ENV = {
    **_GENERAL_TEST_ENV,
    "CAYU_REQUIRE_PLAYWRIGHT_CONTAINMENT": "1",
}
_SPECIALIST_LANES = {
    "stress-process": ("stress or process", 1, 1),
    "postgres-conformance-1": ("postgres and not (stress or process)", 2, 1),
    "postgres-conformance-2": ("postgres and not (stress or process)", 2, 2),
}
_SQLITE_PYTHON_VERSIONS = ("3.11", "3.12", "3.13", "3.14")
_MINIMUM_NODE_VERSION = (22, 18, 0)

_SQLITE_CANCELLATION_TESTS = (
    "tests/core/test_sqlite_session_store.py::"
    "test_sqlite_off_thread_writer_retains_connection_ownership_during_cancellation",
    "tests/core/test_sqlite_session_store.py::"
    "test_sqlite_pending_cancellation_retains_off_thread_connection_ownership",
    "tests/core/test_sqlite_session_store.py::"
    "test_sqlite_task_sweep_cannot_end_off_thread_connection_ownership",
    "tests/core/test_sqlite_session_store.py::"
    "test_sqlite_off_thread_reader_retains_connection_ownership_during_cancellation",
    "tests/core/test_sqlite_session_store.py::"
    "test_sqlite_off_thread_worker_failure_and_cancellation_remain_observable",
    "tests/core/test_sqlite_session_store.py::"
    "test_sqlite_in_memory_read_cancellation_serializes_shared_writer_connection",
    "tests/core/test_runtime.py::"
    "test_cancelled_sqlite_factory_checkpoint_preserves_committed_allocation",
    "tests/core/test_workspace_mutation_receipts.py::"
    "test_final_workspace_observer_restores_caller_cancellation_requests",
    "tests/core/test_workspace_mutation_receipts.py::"
    "test_delegated_stream_close_counts_checkpoint_cancellation_once",
    "tests/core/test_workspace_mutation_receipts.py::"
    "test_delegated_stream_close_distinguishes_restored_and_late_cancellation",
)

_RELEASE_OWNED_PATHS = (
    "dist",
    ".release-venv",
    ".release-all-venv",
    ".release-server-venv",
    ".release-sdist-venv",
)


@dataclass(frozen=True)
class VerificationScope:
    dashboard: bool
    release_artifacts: bool
    sqlite_cancellation: bool


@dataclass(frozen=True)
class PackageStep:
    name: str
    command: str
    publishing_only: bool


@dataclass(frozen=True)
class CommandEvidence:
    label: str
    command: str
    cwd: str
    status: str
    returncode: int | None
    duration_seconds: float
    output_tail: tuple[str, ...]


class StepFailed(RuntimeError):
    pass


def _raise_open_file_limit(minimum: int = 8192) -> tuple[int, int]:
    """Give macOS test shards the descriptor headroom available in hosted CI."""

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = max(soft, minimum)
    if hard != resource.RLIM_INFINITY:
        target = min(target, hard)
    if target > soft:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    return soft, resource.getrlimit(resource.RLIMIT_NOFILE)[0]


def _package_temporary_parent() -> Path | None:
    """Avoid the /var and /tmp symlink aliases rejected by macOS path guards."""

    if platform.system() == "Darwin":
        return Path("/private/tmp")
    return None


def _validate_node_environment(runner: LocalCiRunner) -> None:
    if not runner.dry_run:
        for command in ("node", "npm"):
            if shutil.which(command) is None:
                raise RuntimeError(f"local CI requires {command} on PATH")
        raw_version = _capture(("node", "--version"), cwd=runner.root)
        match = re.fullmatch(r"v(\d+)\.(\d+)\.(\d+)", raw_version)
        if match is None:
            raise RuntimeError(f"could not parse Node.js version: {raw_version!r}")
        version = tuple(int(part) for part in match.groups())
        if version < _MINIMUM_NODE_VERSION:
            minimum = ".".join(str(part) for part in _MINIMUM_NODE_VERSION)
            raise RuntimeError(f"local CI requires Node.js >= {minimum}; found {raw_version}")
    runner.run("Node.js version", ("node", "--version"))
    runner.run("npm version", ("npm", "--version"))


class LocalCiRunner:
    def __init__(
        self,
        *,
        root: Path,
        dry_run: bool,
        keep_going: bool,
    ) -> None:
        self.root = root
        self.dry_run = dry_run
        self.keep_going = keep_going
        self.evidence: list[CommandEvidence] = []
        self.failed = False

    def skip(self, label: str, command: str, *, reason: str) -> None:
        print(f"\n==> {label}")
        print(f"<== SKIP ({reason})")
        self.evidence.append(
            CommandEvidence(
                label=label,
                command=command,
                cwd=".",
                status="SKIP",
                returncode=None,
                duration_seconds=0.0,
                output_tail=(reason,),
            )
        )

    def run(
        self,
        label: str,
        command: Sequence[str] | str,
        *,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        shell: bool = False,
        show_env: bool = True,
        proof_command: str | None = None,
        output_tail_lines: int = 80,
    ) -> bool:
        cwd = cwd or self.root
        command_text = command if isinstance(command, str) else shlex.join(command)
        recorded_command = proof_command or command_text
        if env and show_env:
            assignments = " ".join(
                f"{name}={shlex.quote(value)}" for name, value in sorted(env.items())
            )
            recorded_command = f"env {assignments} {recorded_command}"

        relative_cwd = _display_path(cwd, self.root)
        print(f"\n==> {label}")
        print(f"[{relative_cwd}] $ {recorded_command}")
        if self.dry_run:
            self.evidence.append(
                CommandEvidence(
                    label=label,
                    command=recorded_command,
                    cwd=relative_cwd,
                    status="PLANNED",
                    returncode=None,
                    duration_seconds=0.0,
                    output_tail=(),
                )
            )
            return True

        process_env = os.environ.copy()
        if env:
            process_env.update(env)
        argv: Sequence[str]
        if shell:
            argv = ("/bin/bash", "-eo", "pipefail", "-c", command_text)
        elif isinstance(command, str):
            raise TypeError("string commands require shell=True")
        else:
            argv = command

        started = time.monotonic()
        output_tail: deque[str] = deque(maxlen=output_tail_lines)
        process = subprocess.Popen(
            argv,
            cwd=cwd,
            env=process_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            output_tail.append(line.rstrip("\n"))
        returncode = process.wait()
        duration = time.monotonic() - started
        status = "PASS" if returncode == 0 else "FAIL"
        self.evidence.append(
            CommandEvidence(
                label=label,
                command=recorded_command,
                cwd=relative_cwd,
                status=status,
                returncode=returncode,
                duration_seconds=duration,
                output_tail=tuple(output_tail),
            )
        )
        print(f"<== {status} ({duration:.1f}s)")
        if returncode != 0:
            self.failed = True
            if not self.keep_going:
                raise StepFailed(f"{label} failed with exit code {returncode}")
            return False
        return True


def _display_path(path: Path, root: Path) -> str:
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError:
        return str(path)
    return "." if relative == Path(".") else str(relative)


def _capture(argv: Sequence[str], *, cwd: Path) -> str:
    completed = subprocess.run(
        argv,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _git(root: Path, *args: str) -> str:
    return _capture(("git", *args), cwd=root)


@contextmanager
def _isolated_execution_worktree(source_root: Path, head_sha: str) -> Iterator[Path]:
    with tempfile.TemporaryDirectory(
        prefix="cayu-local-ci-worktree-",
        dir=_package_temporary_parent(),
    ) as parent:
        checkout = Path(parent) / "checkout"
        _capture(
            ("git", "worktree", "add", "--detach", str(checkout), head_sha),
            cwd=source_root,
        )
        try:
            yield checkout
        finally:
            completed = subprocess.run(
                ("git", "worktree", "remove", "--force", str(checkout)),
                cwd=source_root,
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    "could not remove isolated CI worktree: "
                    + (completed.stderr.strip() or completed.stdout.strip())
                )


def _resolve_scope(root: Path, base_sha: str, head_sha: str) -> VerificationScope:
    output = _capture(
        (
            sys.executable,
            "scripts/select_ci_jobs.py",
            "--base-sha",
            base_sha,
            "--head-sha",
            head_sha,
        ),
        cwd=root,
    )
    values: dict[str, bool] = {}
    for line in output.splitlines():
        name, separator, raw_value = line.partition("=")
        if not separator or raw_value not in {"true", "false"}:
            raise RuntimeError(f"unexpected CI scope output: {line!r}")
        values[name] = raw_value == "true"
    expected = {"dashboard", "release_artifacts", "sqlite_cancellation"}
    if values.keys() != expected:
        raise RuntimeError(f"unexpected CI scope keys: {sorted(values)}")
    return VerificationScope(**values)


def _load_package_steps(manifest_path: Path) -> list[PackageStep]:
    """Load the canonical package lane without depending on a YAML package."""

    lines = manifest_path.read_text(encoding="utf-8").splitlines()
    if lines[:2] != ["version: 1", "steps:"]:
        raise RuntimeError("package CI manifest must start with version 1 and steps")
    step_lines = lines[2:]
    step_starts = [index for index, line in enumerate(step_lines) if line.startswith("  - ")]
    if not step_starts:
        raise RuntimeError("package CI manifest contains no steps")
    steps: list[PackageStep] = []
    for position, step_start in enumerate(step_starts):
        step_end = step_starts[position + 1] if position + 1 < len(step_starts) else len(step_lines)
        segment = step_lines[step_start:step_end]
        name = _manifest_scalar(segment[0], "name", list_item=True)
        command = _extract_manifest_run_scalar(segment)
        raw_publishing = next(
            (
                value
                for line in segment
                if (value := _manifest_scalar(line, "publishing")) is not None
            ),
            "false",
        )
        if not name or command is None:
            raise RuntimeError("every package CI step requires name and run")
        if raw_publishing not in {"true", "false"}:
            raise RuntimeError(
                f"package CI step {name!r} has invalid publishing value {raw_publishing!r}"
            )
        steps.append(
            PackageStep(
                name=name,
                command=command,
                publishing_only=raw_publishing == "true",
            )
        )

    required_names = {
        "Verify immutable release state",
        "Check reproducible archives",
        "Check built-wheel authoring contract",
        "Check built-wheel secure public-service contract",
        "Check installed CLI version",
    }
    missing = required_names - {step.name for step in steps}
    if missing:
        raise RuntimeError(f"package CI manifest missed required steps: {sorted(missing)}")
    return steps


def _manifest_scalar(line: str, key: str, *, list_item: bool = False) -> str | None:
    prefix = f"  - {key}:" if list_item else f"    {key}:"
    if line.startswith(prefix):
        return line[len(prefix) :].strip().strip("\"'")
    return None


def _extract_manifest_run_scalar(segment: Sequence[str]) -> str | None:
    for index, line in enumerate(segment):
        raw_value = _manifest_scalar(line, "run")
        if raw_value is None:
            continue
        if raw_value not in {"|", "|-", ">", ">-"}:
            return raw_value
        key_indent = len(line) - len(line.lstrip())
        content_indent = key_indent + 2
        block_lines: list[str] = []
        for block_line in segment[index + 1 :]:
            if not block_line.strip():
                block_lines.append("")
                continue
            indentation = len(block_line) - len(block_line.lstrip())
            if indentation < content_indent:
                break
            block_lines.append(block_line[content_indent:])
        block = "\n".join(block_lines).rstrip()
        if raw_value.startswith(">"):
            return " ".join(part.strip() for part in block.splitlines() if part.strip())
        return block
    return None


def _ensure_exact_head(root: Path, requested_head: str) -> tuple[str, tuple[str, ...]]:
    current_sha = _git(root, "rev-parse", "HEAD")
    head_sha = _git(root, "rev-parse", requested_head)
    if current_sha != head_sha:
        raise RuntimeError(
            f"requested head {head_sha} is not checked out; current HEAD is {current_sha}"
        )
    unstaged = subprocess.run(("git", "diff", "--quiet"), cwd=root).returncode
    staged = subprocess.run(("git", "diff", "--cached", "--quiet"), cwd=root).returncode
    if unstaged != 0 or staged != 0:
        raise RuntimeError("tracked worktree or index changes exist; commit or isolate them first")
    status = _git(root, "status", "--short")
    untracked = tuple(line for line in status.splitlines() if line.startswith("?? "))
    return head_sha, untracked


def _run_static(runner: LocalCiRunner) -> None:
    runner.run("Install Python 3.11", ("uv", "python", "install", "3.11"))
    runner.run(
        "Sync static-check environment",
        (
            "uv",
            "sync",
            "--frozen",
            "--extra",
            "dev",
            "--extra",
            "server",
            "--extra",
            "browser",
            "--python",
            "3.11",
        ),
    )
    for label, command in (
        (
            "Ruff lint",
            (
                "uv",
                "run",
                "--no-sync",
                "ruff",
                "check",
                "src/",
                "tests/",
                "examples/",
                "scripts/",
                "maintenance/",
            ),
        ),
        (
            "Ruff format",
            (
                "uv",
                "run",
                "--no-sync",
                "ruff",
                "format",
                "--check",
                "src/",
                "tests/",
                "examples/",
                "scripts/",
                "maintenance/",
            ),
        ),
        (
            "Ty type check",
            ("uv", "run", "--no-sync", "ty", "check", "src/cayu", "examples", "maintenance"),
        ),
        (
            "Sidecar manifest drift",
            (
                "uv",
                "run",
                "--no-sync",
                "python",
                "scripts/generate_sidecar_manifest.py",
                "--check",
            ),
        ),
        (
            "Dashboard source-bundle drift",
            (
                "uv",
                "run",
                "--no-sync",
                "python",
                "scripts/build_dashboard_source_bundle.py",
                "--check",
            ),
        ),
        ("Lockfile consistency", ("uv", "lock", "--check")),
    ):
        runner.run(label, command)


def _run_docker_prerequisite(runner: LocalCiRunner) -> None:
    runner.run(
        "Docker prerequisite",
        ("docker", "info", "--format", "{{.ServerVersion}} {{.OSType}}/{{.Architecture}}"),
    )


def _sync_python_test_environment(runner: LocalCiRunner, *, browser: bool) -> None:
    runner.run("Install Python 3.14", ("uv", "python", "install", "3.14"))
    extras = ["--extra", "dev", "--extra", "server"]
    if browser:
        extras.extend(("--extra", "browser"))
    runner.run(
        "Sync Python 3.14 test environment" + (" with browser support" if browser else ""),
        (
            "uv",
            "sync",
            "--frozen",
            *extras,
            "--python",
            "3.14",
        ),
    )


def _run_general_shard(runner: LocalCiRunner, shard: int) -> None:
    if shard not in range(1, 7):
        raise ValueError("general shard must be between 1 and 6")
    runner.run(
        f"Python 3.14 general shard {shard}/6",
        (
            "uv",
            "run",
            "--no-sync",
            "pytest",
            "-q",
            "-m",
            "not (stress or process or postgres)",
            "--splits",
            "6",
            "--group",
            str(shard),
            "--splitting-algorithm",
            "least_duration",
            "--durations=20",
        ),
        env=_GENERAL_TEST_ENV,
    )


def _install_playwright_chromium(runner: LocalCiRunner) -> None:
    playwright_command = ["uv", "run", "--no-sync", "playwright", "install"]
    if platform.system() == "Linux":
        playwright_command.append("--with-deps")
    playwright_command.append("chromium")
    runner.run("Install Playwright Chromium", tuple(playwright_command))


def _run_specialist_lane(runner: LocalCiRunner, lane: str) -> None:
    try:
        marker, splits, group = _SPECIALIST_LANES[lane]
    except KeyError as exc:
        raise ValueError(f"unknown specialist lane: {lane}") from exc
    runner.run(
        f"Python 3.14 specialist lane {lane}",
        (
            "uv",
            "run",
            "--no-sync",
            "pytest",
            "-q",
            "-m",
            marker,
            "--splits",
            str(splits),
            "--group",
            str(group),
            "--splitting-algorithm",
            "least_duration",
            "--durations=20",
        ),
        env=_SPECIALIST_TEST_ENV,
    )


def _run_general_shards(runner: LocalCiRunner, *, jobs: int) -> None:
    if jobs not in range(1, 7):
        raise ValueError("local general-shard jobs must be between 1 and 6")
    if runner.dry_run or jobs == 1:
        for shard in range(1, 7):
            _run_general_shard(runner, shard)
        return

    shard_runners = {
        shard: LocalCiRunner(root=runner.root, dry_run=False, keep_going=False)
        for shard in range(1, 7)
    }
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=jobs, thread_name_prefix="cayu-ci-shard") as executor:
        futures = {
            shard: executor.submit(_run_general_shard, shard_runner, shard)
            for shard, shard_runner in shard_runners.items()
        }
        for shard, future in futures.items():
            try:
                future.result()
            except (OSError, RuntimeError, ValueError) as exc:
                failures.append(f"shard {shard}: {exc}")

    for shard_runner in shard_runners.values():
        runner.evidence.extend(shard_runner.evidence)
        runner.failed = runner.failed or shard_runner.failed
    if failures:
        runner.failed = True
        if not runner.keep_going:
            raise StepFailed("general shard failures: " + "; ".join(failures))


def _run_python_suite(runner: LocalCiRunner, *, jobs: int) -> None:
    _run_docker_prerequisite(runner)
    _sync_python_test_environment(runner, browser=False)
    _run_general_shards(runner, jobs=jobs)
    _sync_python_test_environment(runner, browser=True)
    _install_playwright_chromium(runner)
    for lane in _SPECIALIST_LANES:
        _run_specialist_lane(runner, lane)


def _run_sqlite_cancellation_version(runner: LocalCiRunner, python_version: str) -> None:
    if python_version not in _SQLITE_PYTHON_VERSIONS:
        raise ValueError(f"unsupported SQLite cancellation Python: {python_version}")
    runner.run(
        f"Install Python {python_version} for SQLite cancellation",
        ("uv", "python", "install", python_version),
    )
    runner.run(
        f"Sync Python {python_version} SQLite environment",
        (
            "uv",
            "sync",
            "--frozen",
            "--extra",
            "dev",
            "--extra",
            "server",
            "--python",
            python_version,
        ),
    )
    for attempt in range(1, 4):
        command = (
            "uv",
            "run",
            "--no-sync",
            "pytest",
            "-q",
            *_SQLITE_CANCELLATION_TESTS,
        )
        runner.run(
            f"SQLite cancellation Python {python_version}, attempt {attempt}/3",
            command,
            proof_command=(
                "uv run --no-sync pytest -q <10 canonical SQLite cancellation nodes> "
                f"# sha256={hashlib.sha256(shlex.join(command).encode()).hexdigest()}"
            ),
        )


def _run_sqlite_cancellation(runner: LocalCiRunner) -> None:
    for python_version in _SQLITE_PYTHON_VERSIONS:
        _run_sqlite_cancellation_version(runner, python_version)


def _run_dashboard(runner: LocalCiRunner) -> None:
    runner.run("Install Python 3.11 for dashboard", ("uv", "python", "install", "3.11"))
    runner.run(
        "Sync dashboard Python environment",
        (
            "uv",
            "sync",
            "--frozen",
            "--extra",
            "dev",
            "--extra",
            "server",
            "--python",
            "3.11",
        ),
    )
    dashboard = runner.root / "dashboard"
    _validate_node_environment(runner)
    for label, npm_args in (
        ("Dashboard install", ("ci",)),
        ("Dashboard lint", ("run", "lint")),
        ("Dashboard tests", ("run", "test")),
        ("Dashboard typecheck", ("run", "typecheck")),
        ("Dashboard API drift", ("run", "check:api:repo")),
        ("Dashboard package build", ("run", "build:package")),
    ):
        runner.run(label, ("npm", *npm_args), cwd=dashboard)
    runner.run(
        "Packaged dashboard assets unchanged",
        """
status="$(git status --porcelain -- src/cayu/server/dashboard)"
if test -n "$status"; then
  echo "$status"
  git diff --stat -- src/cayu/server/dashboard
  exit 1
fi
""".strip(),
        shell=True,
    )


def _prepare_package_environment(runner: LocalCiRunner) -> None:
    runner.run("Install Python 3.11 for package checks", ("uv", "python", "install", "3.11"))
    runner.run(
        "Sync Python 3.11 package environment",
        (
            "uv",
            "sync",
            "--frozen",
            "--extra",
            "dev",
            "--extra",
            "server",
            "--extra",
            "browser",
            "--python",
            "3.11",
        ),
    )
    _validate_node_environment(runner)
    runner.run("Verify ripgrep prerequisite", ("rg", "--version"))
    _run_docker_prerequisite(runner)


def _run_release_artifacts(
    runner: LocalCiRunner,
    temporary_root: Path,
    *,
    publishing: bool,
) -> None:
    # macOS exposes /tmp and /var as symlinks. Package security checks reject
    # traversing either alias, so pass their canonical /private/... destination.
    temporary_root = temporary_root.resolve()
    existing = [
        name
        for name in _RELEASE_OWNED_PATHS
        if (runner.root / name).exists() or (runner.root / name).is_symlink()
    ]
    if existing and not runner.dry_run:
        raise RuntimeError(
            "release checks require these script-owned paths to be absent: " + ", ".join(existing)
        )
    if not runner.dry_run and shutil.which("rg") is None:
        raise RuntimeError("release checks require rg on PATH")

    github_ref = "refs/heads/local-pr"
    github_ref_name = "local-pr"
    if publishing and not runner.dry_run:
        tags = tuple(
            tag
            for tag in _git(runner.root, "tag", "--points-at", "HEAD", "--list", "v*").splitlines()
            if tag
        )
        if len(tags) != 1:
            raise RuntimeError(
                "publishing verification requires HEAD to have exactly one v* tag; "
                f"found {len(tags)}"
            )
        github_ref_name = tags[0]
        github_ref = f"refs/tags/{github_ref_name}"
    github_output = str(temporary_root / "github-output")
    if os.environ.get("GITHUB_ACTIONS") == "true" and os.environ.get("GITHUB_OUTPUT"):
        github_output = os.environ["GITHUB_OUTPUT"]
    release_env = {
        "GITHUB_WORKSPACE": str(runner.root),
        "GITHUB_REF": github_ref,
        "GITHUB_REF_NAME": github_ref_name,
        "GITHUB_OUTPUT": github_output,
        "RUNNER_TEMP": str(temporary_root),
        "TMPDIR": str(temporary_root),
        "PATH": os.environ["PATH"],
    }
    steps = _load_package_steps(runner.root / "scripts/package_ci_steps.yml")
    preserve_distribution = publishing and os.environ.get("GITHUB_ACTIONS") == "true"
    try:
        _prepare_package_environment(runner)
        for step in steps:
            command = step.command
            if platform.system() != "Linux":
                command = command.replace(
                    "playwright install --with-deps chromium",
                    "playwright install chromium",
                )
            proof_command = (
                "bash -eo pipefail <canonical package step "
                f"{shlex.quote(step.name)}> # sha256="
                f"{hashlib.sha256(command.encode()).hexdigest()}"
            )
            if step.publishing_only and not publishing:
                runner.skip(
                    f"Publishing only: {step.name}",
                    proof_command,
                    reason="pass --publishing only when preparing a publication",
                )
                continue
            runner.run(
                f"Release artifacts: {step.name}",
                command,
                shell=True,
                env=release_env,
                show_env=False,
                proof_command=proof_command,
                output_tail_lines=12,
            )
    finally:
        if not runner.dry_run:
            for name in _RELEASE_OWNED_PATHS:
                if preserve_distribution and name == "dist":
                    continue
                path = runner.root / name
                if path.is_symlink() or path.is_file():
                    path.unlink(missing_ok=True)
                elif path.is_dir():
                    shutil.rmtree(path)


def _write_proof(
    path: Path,
    *,
    status: str,
    started_at: datetime,
    finished_at: datetime,
    root: Path,
    base_sha: str,
    head_sha: str,
    merge_base: str,
    changed_paths: Sequence[str],
    scope: VerificationScope,
    source_untracked: Sequence[str],
    execution_untracked: Sequence[str],
    open_file_limits: tuple[int, int],
    jobs: int,
    publishing: bool,
    evidence: Sequence[CommandEvidence],
    error: str | None,
) -> None:
    status_counts = {
        status_name: sum(item.status == status_name for item in evidence)
        for status_name in ("PASS", "FAIL", "SKIP", "PLANNED")
    }
    lines = [
        "# Local PR CI proof",
        "",
        f"- Status: **{status}**",
        f"- Commit tested: `{head_sha}`",
        f"- Requested base: `{base_sha}`",
        f"- Merge base: `{merge_base}`",
        f"- Source repository: `{root}`",
        "- Execution checkout: fresh detached worktree at the tested commit",
        f"- Platform: `{platform.system()} {platform.machine()}`",
        "- Open-file soft limit: "
        f"`{open_file_limits[0]}` before runner setup; `{open_file_limits[1]}` during tests",
        f"- Started: `{started_at.isoformat()}`",
        f"- Finished: `{finished_at.isoformat()}`",
        f"- Execution: GitHub's six general Python shards run with `{jobs}` local "
        "worker(s); specialist lanes remain serial; this does not claim the Ubuntu "
        "runner or its separate-machine topology",
        "- Command counts: "
        f"`{status_counts['PASS']} passed`, `{status_counts['FAIL']} failed`, "
        f"`{status_counts['SKIP']} skipped`, `{status_counts['PLANNED']} planned`",
        "",
        "## Selected scope",
        "",
        f"- `dashboard={str(scope.dashboard).lower()}`",
        f"- `release_artifacts={str(scope.release_artifacts).lower()}`",
        f"- `sqlite_cancellation={str(scope.sqlite_cancellation).lower()}`",
        f"- `publishing={str(publishing).lower()}`",
        "",
        "## Changed paths",
        "",
    ]
    lines.extend(f"- `{path_name}`" for path_name in changed_paths)
    if not changed_paths:
        lines.append("- None")
    lines.extend(("", "## Working-tree evidence", ""))
    lines.append("- Source tracked worktree and index were clean before execution.")
    if source_untracked:
        lines.append(
            "- Source untracked paths were recorded but absent from the isolated execution checkout:"
        )
        lines.extend(f"  - `{item[3:]}`" for item in source_untracked)
    else:
        lines.append("- No source untracked paths were present.")
    if execution_untracked:
        lines.append("- Untracked paths generated inside the isolated checkout during execution:")
        lines.extend(f"  - `{item[3:]}`" for item in execution_untracked)
    else:
        lines.append("- No non-ignored untracked paths were generated during execution.")
    if error:
        lines.extend(("", "## Failure", "", f"`{error}`"))
    lines.extend(("", "## Command results", ""))
    for item in evidence:
        lines.extend(
            (
                f"### {item.status}: {item.label}",
                "",
                f"- Working directory: `{item.cwd}`",
                f"- Exit code: `{item.returncode}`"
                if item.returncode is not None
                else "- Exit code: not run",
                f"- Duration: `{item.duration_seconds:.1f}s`",
                "",
                "```bash",
                item.command,
                "```",
            )
        )
        if item.output_tail:
            lines.extend(("", "Output tail:", "", "````text", *item.output_tail, "````"))
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lane",
        choices=(
            "static",
            "general",
            "specialist",
            "sqlite-cancellation",
            "dashboard",
            "package",
        ),
        help="run one canonical lane (used by GitHub Actions)",
    )
    parser.add_argument("--shard", type=int, help="general lane shard, from 1 through 6")
    parser.add_argument(
        "--specialist-lane",
        choices=tuple(_SPECIALIST_LANES),
        help="specialist lane identity",
    )
    parser.add_argument(
        "--python-version",
        choices=_SQLITE_PYTHON_VERSIONS,
        help="SQLite cancellation lane Python version",
    )
    parser.add_argument(
        "--base",
        default="origin/main",
        help="PR base ref or SHA (default: origin/main; this script never fetches it)",
    )
    parser.add_argument("--head", default="HEAD", help="checked-out PR head ref or SHA")
    parser.add_argument(
        "--proof",
        type=Path,
        help="Markdown proof path (default: /tmp/cayu-local-ci-proof-<sha>.md)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="run every retained PR lane instead of using changed-path selection",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve scope and print commands without executing them",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="continue after command failures and report all observed failures",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        choices=range(1, 7),
        default=2,
        help="concurrent local general shards (default: 2; specialist lanes stay serial)",
    )
    parser.add_argument(
        "--publishing",
        action="store_true",
        help="include immutable release-note and published-version checks",
    )
    return parser.parse_args(argv)


def _fatal(message: str) -> NoReturn:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(2)


def _run_ci_lane(args: argparse.Namespace) -> int:
    root = Path(_capture(("git", "rev-parse", "--show-toplevel"), cwd=Path.cwd()))
    _raise_open_file_limit()
    runner = LocalCiRunner(root=root, dry_run=args.dry_run, keep_going=False)
    try:
        if args.lane == "static":
            _run_static(runner)
        elif args.lane == "general":
            if args.shard is None:
                raise ValueError("--lane general requires --shard")
            _run_docker_prerequisite(runner)
            _sync_python_test_environment(runner, browser=False)
            _run_general_shard(runner, args.shard)
        elif args.lane == "specialist":
            if args.specialist_lane is None:
                raise ValueError("--lane specialist requires --specialist-lane")
            _run_docker_prerequisite(runner)
            _sync_python_test_environment(runner, browser=True)
            if args.specialist_lane == "stress-process":
                _install_playwright_chromium(runner)
            _run_specialist_lane(runner, args.specialist_lane)
        elif args.lane == "sqlite-cancellation":
            if args.python_version is None:
                raise ValueError("--lane sqlite-cancellation requires --python-version")
            _run_sqlite_cancellation_version(runner, args.python_version)
        elif args.lane == "dashboard":
            _run_dashboard(runner)
        elif args.lane == "package":
            with tempfile.TemporaryDirectory(
                prefix="cayu-local-package-ci-",
                dir=_package_temporary_parent(),
            ) as temporary:
                _run_release_artifacts(
                    runner,
                    Path(temporary),
                    publishing=args.publishing,
                )
        else:  # pragma: no cover - argparse owns the closed choice set
            raise ValueError(f"unknown CI lane: {args.lane}")
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 1 if runner.failed else 0


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    args = _parse_args(argv)
    if args.lane is not None:
        return _run_ci_lane(args)
    try:
        source_root = Path(_capture(("git", "rev-parse", "--show-toplevel"), cwd=Path.cwd()))
        head_sha, source_untracked = _ensure_exact_head(source_root, args.head)
        base_sha = _git(source_root, "rev-parse", args.base)
        merge_base = _git(source_root, "merge-base", base_sha, head_sha)
        changed_output = _git(
            source_root,
            "diff",
            "--name-only",
            "--no-renames",
            f"{base_sha}...{head_sha}",
        )
        changed_paths = tuple(line for line in changed_output.splitlines() if line)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        _fatal(str(exc))

    proof_path = args.proof
    if proof_path is None:
        proof_path = Path("/tmp") / f"cayu-local-ci-proof-{head_sha[:12]}.md"
    elif not proof_path.is_absolute():
        proof_path = source_root / proof_path

    try:
        open_file_limits = _raise_open_file_limit()
    except (OSError, ValueError) as exc:
        _fatal(f"could not raise the open-file limit for CI parity: {exc}")

    started_at = datetime.now(timezone.utc)  # noqa: UP017 - system Python 3.9
    try:
        with _isolated_execution_worktree(source_root, head_sha) as execution_root:
            _, initial_execution_untracked = _ensure_exact_head(execution_root, head_sha)
            if initial_execution_untracked:
                raise RuntimeError("isolated CI worktree unexpectedly contains untracked paths")
            scope = (
                VerificationScope(True, True, True)
                if args.all
                else _resolve_scope(execution_root, base_sha, head_sha)
            )
            if args.publishing and not scope.release_artifacts:
                scope = VerificationScope(
                    dashboard=scope.dashboard,
                    release_artifacts=True,
                    sqlite_cancellation=scope.sqlite_cancellation,
                )

            print(f"Repository: {source_root}")
            print(f"Execution:  isolated detached worktree at {head_sha}")
            print(f"Head:       {head_sha}")
            print(f"Base:       {base_sha}")
            print(
                "Scope:      "
                f"dashboard={scope.dashboard} "
                f"release_artifacts={scope.release_artifacts} "
                f"sqlite_cancellation={scope.sqlite_cancellation} "
                f"publishing={args.publishing} jobs={args.jobs}"
            )
            print(f"Open files: soft limit {open_file_limits[0]} -> {open_file_limits[1]}")
            if source_untracked:
                print("Source untracked paths (excluded by isolated execution):")
                for item in source_untracked:
                    print(f"  {item}")

            runner = LocalCiRunner(
                root=execution_root,
                dry_run=args.dry_run,
                keep_going=args.keep_going,
            )
            error: str | None = None
            interrupted = False
            execution_untracked: tuple[str, ...] = ()
            with tempfile.TemporaryDirectory(
                prefix="cayu-local-pr-ci-",
                dir=_package_temporary_parent(),
            ) as temporary_directory:
                try:
                    _run_static(runner)
                    _run_python_suite(runner, jobs=args.jobs)
                    if scope.sqlite_cancellation:
                        _run_sqlite_cancellation(runner)
                    if scope.dashboard:
                        _run_dashboard(runner)
                    if scope.release_artifacts:
                        _run_release_artifacts(
                            runner,
                            Path(temporary_directory),
                            publishing=args.publishing,
                        )
                    runner.run(
                        "Exact head and tracked tree remain clean",
                        (
                            f'test "$(git rev-parse HEAD)" = "{head_sha}"\n'
                            "git diff --exit-code\n"
                            "git diff --cached --exit-code"
                        ),
                        shell=True,
                    )
                    _, execution_untracked = _ensure_exact_head(execution_root, head_sha)
                except KeyboardInterrupt:
                    interrupted = True
                    error = "interrupted by user"
                except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
                    runner.failed = True
                    error = str(exc)

            if interrupted:
                status = "INTERRUPTED"
            elif args.dry_run:
                status = "DRY RUN"
            elif runner.failed:
                status = "FAIL"
            else:
                status = "PASS"
            finished_at = datetime.now(timezone.utc)  # noqa: UP017 - system Python 3.9
            _write_proof(
                proof_path,
                status=status,
                started_at=started_at,
                finished_at=finished_at,
                root=source_root,
                base_sha=base_sha,
                head_sha=head_sha,
                merge_base=merge_base,
                changed_paths=changed_paths,
                scope=scope,
                source_untracked=source_untracked,
                execution_untracked=execution_untracked,
                open_file_limits=open_file_limits,
                jobs=args.jobs,
                publishing=args.publishing,
                evidence=runner.evidence,
                error=error,
            )
            print(f"\nProof: {proof_path}")
            print(f"Result: {status}")
            if interrupted:
                return 130
            return 1 if runner.failed else 0
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        _fatal(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
