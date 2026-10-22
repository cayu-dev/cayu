"""Real Docker proof that a command timeout stops its tree and keeps the container."""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
from uuid import uuid4

import pytest

from cayu.runners.base import ExecCommand
from cayu.runners.docker import DOCKER_COMMAND_STATE_DIR, DockerRunner
from cayu.runners.docker_workload import DockerImageIdentity, DockerWorkloadRestrictions

pytestmark = pytest.mark.process

_REQUIRE_DOCKER_RUNNER_ENV_VAR = "CAYU_REQUIRE_DOCKER_RUNNER"
_IMAGE = os.environ.get("CAYU_DOCKER_PYTHON_LIVE_IMAGE", "python:3.12-slim")
_PIDS = "/tmp/cayu-tree-pids"

# A foreground child, a setsid daemon that leaves the process group, and a
# double-forked orphan. Each records its PID before it blocks.
_TREE_SCRIPT = (
    f"setsid sh -c 'echo $$ >> {_PIDS}; exec sleep 300' & "
    f"(sh -c 'echo $$ >> {_PIDS}; exec sleep 300' &); "
    f"echo $$ >> {_PIDS}; "
    f'while test "$(wc -l < {_PIDS})" -lt 3; do sleep 0.05; done; '
    "sleep 300"
)


def _unavailable(reason: str) -> None:
    if os.environ.get(_REQUIRE_DOCKER_RUNNER_ENV_VAR):
        pytest.fail(reason)
    pytest.skip(reason)


def _docker_and_image() -> tuple[str, DockerImageIdentity]:
    docker = os.environ.get("CAYU_DOCKER_PATH") or shutil.which("docker")
    if docker is None:
        _unavailable("docker CLI not found")
    assert docker is not None
    try:
        subprocess.run(
            [docker, "info"], check=True, capture_output=True, timeout=20, stdin=subprocess.DEVNULL
        )
        inspected = subprocess.run(
            [docker, "image", "inspect", "--format", "{{.Id}}", _IMAGE],
            capture_output=True,
            text=True,
            timeout=20,
            stdin=subprocess.DEVNULL,
        )
        if inspected.returncode != 0:
            subprocess.run([docker, "pull", _IMAGE], check=True, capture_output=True, timeout=300)
            inspected = subprocess.run(
                [docker, "image", "inspect", "--format", "{{.Id}}", _IMAGE],
                check=True,
                capture_output=True,
                text=True,
                timeout=20,
            )
    except Exception as exc:
        _unavailable(f"Docker or {_IMAGE} unavailable: {exc}")
    return docker, DockerImageIdentity(reference=_IMAGE, content_digest=inspected.stdout.strip())


def _container_exists(docker: str, container_id: str) -> bool:
    return (
        subprocess.run(
            [docker, "container", "inspect", container_id],
            capture_output=True,
            timeout=20,
            stdin=subprocess.DEVNULL,
        ).returncode
        == 0
    )


async def _strict_runner(docker: str, image_identity: DockerImageIdentity) -> DockerRunner:
    return await DockerRunner.create(
        f"cayu-tree-cleanup-live-{uuid4().hex[:12]}",
        image=image_identity.reference,
        image_identity=image_identity,
        # Read-only root, nonroot user, all capabilities dropped,
        # no-new-privileges, and pids/memory limits.
        workload_restrictions=DockerWorkloadRestrictions(),
        required_executables=("python3", "setsid", "sh", "sleep"),
        network="none",
        replace=False,
        close_action="remove",
        credential_mode="trusted_tool",
        allow_raw_secret_env=False,
        cancellation_cleanup="command_or_sandbox",
        timeout_cleanup="command_or_sandbox",
        default_cwd="/workspace",
        docker_path=docker,
    )


async def _recorded_pids_are_gone(runner: DockerRunner) -> None:
    observed = await runner.exec(
        ExecCommand.bash(
            f'test "$(wc -l < {_PIDS})" -eq 3 || exit 2; '
            f'while read pid; do test -e "/proc/$pid" && exit 3; done < {_PIDS}; '
            f"rm -f {_PIDS}"
        ),
        timeout_s=10,
    )
    assert (observed.exit_code, observed.stderr) == (0, "")


@pytest.mark.parametrize("kind", ["shell", "process"])
def test_timed_out_command_tree_is_stopped_and_container_is_kept(kind: str) -> None:
    docker, image_identity = _docker_and_image()
    command = (
        ExecCommand.bash(_TREE_SCRIPT)
        if kind == "shell"
        else ExecCommand.process("sh", "-c", _TREE_SCRIPT)
    )

    async def run() -> None:
        runner = await _strict_runner(docker, image_identity)
        try:
            written = await runner.exec(
                ExecCommand.bash("printf kept > /workspace/marker"), timeout_s=10
            )
            assert written.exit_code == 0

            timed_out = await runner.exec(command, timeout_s=2)

            assert timed_out.timed_out is True
            assert [(item["action"], item["status"]) for item in timed_out.artifacts] == [
                ("kill_command", "completed")
            ]
            assert runner.lifecycle_state == "reusable"
            assert runner.container_id is not None
            assert _container_exists(docker, runner.container_id)
            await _recorded_pids_are_gone(runner)
            kept = await runner.exec(ExecCommand.process("cat", "/workspace/marker"))
            assert (kept.exit_code, kept.stdout) == (0, "kept")
            claim = runner.execution_capability_evidence().claim_for("confirmed_cancellation")
            assert claim is not None
            assert {detail.name: detail.value for detail in claim.adapter_details} == {
                "container_cleanup": True,
                "command_tree_cleanup": True,
                "exact_container_owner": True,
            }
        finally:
            await runner.close()
        assert runner.container_id is not None
        assert not _container_exists(docker, runner.container_id)

    asyncio.run(run())


def test_supervised_shell_keeps_the_environment_it_was_given() -> None:
    docker, image_identity = _docker_and_image()

    async def run() -> None:
        runner = await _strict_runner(docker, image_identity)
        try:
            # Under the C locale CPython coerces LC_CTYPE in its own
            # environment; the supervised shell must not inherit that.
            observed = await runner.exec(
                ExecCommand.bash('printf "%s|" "${LC_CTYPE-unset}"; printf "\\303\\251" | wc -m'),
                env={"LANG": "C"},
                timeout_s=10,
            )
            assert observed.exit_code == 0, observed.stderr
            assert observed.stdout.split() == ["unset|2"]
        finally:
            await runner.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    ("script", "environment", "expected"),
    [
        ("while :; do printf x; done | head -c 1", {}, "x"),
        ('printf "%s" "$PATH"', {"PATH": "/usr/bin:/bin"}, "/usr/bin:/bin"),
    ],
)
def test_supervised_shell_preserves_signals_and_command_path(script, environment, expected):
    docker, image_identity = _docker_and_image()

    async def run() -> None:
        runner = await _strict_runner(docker, image_identity)
        try:
            for _ in range(2):
                result = await runner.exec(ExecCommand.bash(script), env=environment, timeout_s=3)
                assert not result.timed_out
                assert result.exit_code == 0, result.stderr
                assert result.stdout == expected
        finally:
            await runner.close()

    asyncio.run(run())


def test_unconfirmed_command_tree_falls_back_to_container_removal() -> None:
    docker, image_identity = _docker_and_image()

    async def run() -> None:
        runner = await _strict_runner(docker, image_identity)
        try:
            # Removing the supervisor's state file leaves cleanup no proof
            # that the tree exited, so the runner must remove the container.
            timed_out = await runner.exec(
                ExecCommand.bash(f"rm -f {DOCKER_COMMAND_STATE_DIR}/*.pid; sleep 300"),
                timeout_s=2,
            )

            assert timed_out.timed_out is True
            command_cleanup, container_cleanup = timed_out.artifacts
            assert command_cleanup["action"] == "kill_command"
            # Retried helpers can exhaust the phase bound on a loaded host.
            assert command_cleanup["status"] in {"failed", "timeout"}
            assert (container_cleanup["action"], container_cleanup["status"]) == (
                "kill_sandbox",
                "completed",
            )
            assert runner.lifecycle_state == "closed"
            assert runner.container_id is not None
            assert not _container_exists(docker, runner.container_id)
        finally:
            await runner.close()

    asyncio.run(run())
