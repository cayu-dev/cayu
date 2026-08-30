"""Opt-in real-Docker proof for admitted Python and non-Python toolchains."""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
from hashlib import sha256
from pathlib import Path
from typing import Literal, cast
from uuid import uuid4

import pytest

from cayu import (
    CommandPolicy,
    CommandPolicyDecision,
    CommandPolicyResult,
    CommandRequest,
    DockerCodingAdmissionProbe,
    DockerCodingCommandAuthority,
    DockerCodingDependencyInput,
    DockerCodingEnvironmentFactory,
    DockerCodingToolchainProfile,
    DockerImageIdentity,
    EnvironmentFactoryReleaseAction,
    EnvironmentFactoryRequest,
    ExecCommand,
    ExecutionProfileBehaviorIdentity,
    LocalWorkspace,
    NamedCheck,
    RunCheckTool,
    RunCommandTool,
    ToolContext,
    WriteFileTool,
)

_LIVE_ENVIRONMENT = "CAYU_RUN_DOCKER_TOOLCHAIN_LIVE"
_PYTHON_BASE = (
    "python:3.12-slim@sha256:7a8b475003c4fe15a2cd4e55e5cfc2f3560bdc9333d624f24cdd6d4340fd7a17"
)
_NODE_BASE = (
    "node:22.18.0-bookworm-slim@sha256:"
    "752ea8a2f758c34002a0461bd9f1cee4f9a3c36d48494586f60ffce1fc708e0e"
)
_DETACHED_PARENT_CODE = """\
import subprocess
import sys

child = subprocess.Popen(
    [sys.argv[2], "-c", "import time; time.sleep(30)", sys.argv[1]],
    stdin=subprocess.DEVNULL,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    start_new_session=True,
)
print(child.pid, flush=True)
"""


class _ExactNamedCheckPolicy(CommandPolicy):
    def __init__(self, argv: tuple[str, ...]) -> None:
        self._argv = argv

    async def evaluate(
        self,
        ctx: ToolContext,
        request: CommandRequest,
    ) -> CommandPolicyResult:
        del ctx
        command = request.command
        exact = (
            command.kind == "process"
            and tuple(command.argv or ()) == self._argv
            and command.shell is None
            and request.cwd is None
            and request.canonical_cwd == "/workspace"
            and request.env is None
            and request.stdin is None
        )
        return CommandPolicyResult(
            decision=(CommandPolicyDecision.ALLOW if exact else CommandPolicyDecision.DENY)
        )


def _require_live_docker() -> str:
    if os.environ.get(_LIVE_ENVIRONMENT) != "1":
        pytest.skip(f"set {_LIVE_ENVIRONMENT}=1 to run real Docker toolchain proof")
    docker = shutil.which("docker")
    if docker is None:
        pytest.fail("Docker CLI is required by the opted-in toolchain live proof")
    probe = subprocess.run(
        [docker, "info", "--format", "{{.ServerVersion}}"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=20,
        check=False,
    )
    if probe.returncode != 0 or not probe.stdout.strip():
        pytest.fail("Docker daemon is required by the opted-in toolchain live proof")
    return docker


def _trusted_fixture_image(
    docker: str,
    root: Path,
    *,
    label: str,
    base_image: str,
    packages: str,
) -> tuple[str, str, Literal["amd64", "arm64"]]:
    tag = f"cayu-toolchain-live-{label}:{uuid4().hex}"
    dockerfile = root / f"Dockerfile.{label}"
    dockerfile.write_text(
        f"FROM {base_image}\n"
        "RUN apt-get update \\\n"
        f"    && apt-get install -y --no-install-recommends {packages} \\\n"
        "    && rm -rf /var/lib/apt/lists/*\n"
        f'ENTRYPOINT ["sh", "-c", "exit {97 if label == "python" else 98}"]\n',
        encoding="utf-8",
    )
    built = subprocess.run(
        [docker, "build", "--file", str(dockerfile), "--tag", tag, str(root)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=300,
        check=False,
    )
    if built.returncode != 0:
        pytest.fail(f"trusted {label} fixture image build failed")
    inspected = subprocess.run(
        [docker, "image", "inspect", "--format", "{{.Id}} {{.Architecture}}", tag],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=20,
        check=False,
    )
    fields = inspected.stdout.decode("ascii", errors="ignore").strip().split()
    if (
        inspected.returncode != 0
        or len(fields) != 2
        or not fields[0].startswith("sha256:")
        or fields[1] not in {"amd64", "arm64"}
    ):
        _remove_fixture_image(docker, tag)
        pytest.fail(f"trusted {label} fixture image identity could not be resolved")
    return tag, fields[0], cast("Literal['amd64', 'arm64']", fields[1])


def _remove_fixture_image(docker: str, tag: str) -> None:
    subprocess.run(
        [docker, "image", "rm", "--force", tag],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=60,
        check=False,
    )


def _profile(
    *,
    label: str,
    image: str,
    digest: str,
    architecture: Literal["amd64", "arm64"],
    executable: str,
    python_executable: str,
    lock_name: str,
    lock_content: bytes,
) -> DockerCodingToolchainProfile:
    version_selector = f"{label}-version"
    check_selector = f"{label}-check"
    return DockerCodingToolchainProfile(
        profile_id=f"live-{label}",
        revision="1",
        image_identity=DockerImageIdentity(reference=image, content_digest=digest),
        platform_architecture=architecture,
        command_authorities=(
            DockerCodingCommandAuthority(
                selector="bounded-timeout",
                revision="1",
                description="Exercise bounded descendant cleanup.",
                exposure="structured_command",
                executable="/usr/bin/sleep",
                allowed_literals=("30",),
                allow_positional_arguments=True,
                min_arguments=1,
                max_arguments=1,
                timeout_seconds=1,
                max_output_bytes=4096,
            ),
            DockerCodingCommandAuthority(
                selector="detached-timeout",
                revision="1",
                description="Exercise detached descendant settlement.",
                exposure="structured_command",
                executable=python_executable,
                fixed_arguments=(
                    "-c",
                    _DETACHED_PARENT_CODE,
                    f"cayu-{label}-detached",
                    python_executable,
                ),
                max_arguments=0,
                timeout_seconds=1,
                max_output_bytes=4096,
            ),
            DockerCodingCommandAuthority(
                selector=check_selector,
                revision="1",
                description=f"Run the admitted {label} named check.",
                exposure="named_check",
                executable=executable,
                fixed_arguments=("--version",),
                max_arguments=0,
                timeout_seconds=20,
                max_output_bytes=4096,
            ),
            DockerCodingCommandAuthority(
                selector=version_selector,
                revision="1",
                description=f"Run the admitted {label} version diagnostic.",
                exposure="structured_command",
                executable=executable,
                fixed_arguments=("--version",),
                max_arguments=0,
                timeout_seconds=20,
                max_output_bytes=4096,
            ),
        ),
        dependency_inputs=(
            DockerCodingDependencyInput(
                path=lock_name,
                content_sha256="sha256:" + sha256(lock_content).hexdigest(),
            ),
        ),
        admission_probes=(
            DockerCodingAdmissionProbe(
                probe_id=f"{label}-version",
                argv=(executable, "--version"),
                timeout_seconds=20,
                max_output_bytes=4096,
            ),
        ),
    )


async def _exercise_profile(
    source: LocalWorkspace,
    profile: DockerCodingToolchainProfile,
    *,
    label: str,
) -> None:
    created = await DockerCodingEnvironmentFactory(
        source_workspace=source,
        toolchain_profile=profile,
    ).create(
        EnvironmentFactoryRequest(
            session_id=f"live-{label}",
            agent_name="agent",
            environment_name="coding",
        )
    )
    runner = created.environment.runner
    binding = created.environment.binding
    assert runner is not None and binding is not None
    try:
        bound = await binding.bind(
            created.environment.workspace,
            runner,
            session_id=f"live-{label}",
            agent_name="agent",
            environment_name="coding",
        )
        assert bound.workspace is not None
        ctx = ToolContext(
            session_id=f"live-{label}",
            agent_name="agent",
            environment_name="coding",
            workspace_id=bound.workspace.id,
            workspace=bound.workspace,
            runner=runner,
        )
        command = await RunCommandTool(toolchain_profile=profile).run(
            ctx,
            {"selector": f"{label}-version"},
        )
        check_authority = profile.command_authority(f"{label}-check")
        assert check_authority is not None
        check_argv = check_authority.command_argv()
        named_check = await RunCheckTool(
            checks=(
                NamedCheck(
                    name=f"{label}-check",
                    description=f"Run the admitted {label} check.",
                    command=ExecCommand.process(*check_argv),
                    timeout_s=20,
                    max_output_bytes=4096,
                    execution_profile_identity=ExecutionProfileBehaviorIdentity(
                        name=f"live.{label}.named_check",
                        behavior_version="1",
                        implementation_version=profile.fingerprint,
                    ),
                ),
            ),
            command_policy=_ExactNamedCheckPolicy(check_argv),
            toolchain_profile=profile,
        ).run(ctx, {"check": f"{label}-check"})
        mutation = await WriteFileTool().run(
            ctx,
            {
                "path": "live-proof.txt",
                "content": f"{label}-copied-back\n",
                "mode": "create",
            },
        )
        assert command.is_error is False
        assert command.structured["toolchain_profile_fingerprint"] == profile.fingerprint
        assert command.structured["toolchain_image_content_digest"] == (
            profile.image_identity.content_digest
        )
        assert named_check.is_error is False
        assert named_check.structured["toolchain_profile_fingerprint"] == profile.fingerprint
        assert mutation.is_error is False
        candidate = runner.execution_admission_candidate()
        network = candidate.evidence.claim_for("deny_by_default_network")
        assert network is not None
        assert network.state == "live_verified"
        assert network.observation == "denied"
        await binding.finalize(bound, outcome="completed")
    finally:
        if created.release is not None:
            await created.release(EnvironmentFactoryReleaseAction.DISCARD)


async def _exercise_timeout_cleanup(
    source: LocalWorkspace,
    profile: DockerCodingToolchainProfile,
    *,
    label: str,
) -> None:
    created = await DockerCodingEnvironmentFactory(
        source_workspace=source,
        toolchain_profile=profile,
    ).create(
        EnvironmentFactoryRequest(
            session_id=f"live-{label}-timeout",
            agent_name="agent",
            environment_name="coding",
        )
    )
    runner = created.environment.runner
    binding = created.environment.binding
    assert runner is not None and binding is not None
    try:
        bound = await binding.bind(
            created.environment.workspace,
            runner,
            session_id=f"live-{label}-timeout",
            agent_name="agent",
            environment_name="coding",
        )
        assert bound.workspace is not None
        detached = await RunCommandTool(toolchain_profile=profile).run(
            ToolContext(
                session_id=f"live-{label}-timeout",
                agent_name="agent",
                environment_name="coding",
                workspace_id=bound.workspace.id,
                workspace=bound.workspace,
                runner=runner,
            ),
            {"selector": "detached-timeout", "args": []},
        )
        assert detached.is_error is True
        assert detached.structured["status"] == "timed_out"
        assert detached.structured["workspace_mutation_settlement"] == "runner_quiescent"
        assert int(detached.structured["stdout"].strip()) > 1
    finally:
        if created.release is not None:
            await created.release(EnvironmentFactoryReleaseAction.DISCARD)


@pytest.mark.parametrize(
    ("label", "base_image", "packages", "executable", "python_executable"),
    (
        (
            "python",
            _PYTHON_BASE,
            "git",
            "/usr/local/bin/python3",
            "/usr/local/bin/python3",
        ),
        ("node", _NODE_BASE, "git python3", "/usr/local/bin/node", "/usr/bin/python3"),
    ),
)
def test_real_docker_admits_python_and_non_python_profiles(
    tmp_path: Path,
    label: str,
    base_image: str,
    packages: str,
    executable: str,
    python_executable: str,
) -> None:
    docker = _require_live_docker()
    tag = ""
    source_root = tmp_path / f"source-{label}"
    source_root.mkdir()
    lock_name = f"{label}.lock"
    lock_content = f"{label}-locked\n".encode()
    (source_root / lock_name).write_bytes(lock_content)
    (source_root / "README.md").write_text(f"{label}\n", encoding="utf-8")
    try:
        tag, digest, architecture = _trusted_fixture_image(
            docker,
            tmp_path,
            label=label,
            base_image=base_image,
            packages=packages,
        )
        profile = _profile(
            label=label,
            image=tag,
            digest=digest,
            architecture=architecture,
            executable=executable,
            python_executable=python_executable,
            lock_name=lock_name,
            lock_content=lock_content,
        )
        source = LocalWorkspace(source_root, workspace_id=f"source-{label}")
        asyncio.run(_exercise_profile(source, profile, label=label))
        assert (source_root / "live-proof.txt").read_text(encoding="utf-8") == (
            f"{label}-copied-back\n"
        )
        asyncio.run(
            _exercise_timeout_cleanup(
                source,
                profile,
                label=label,
            )
        )
    finally:
        if tag:
            _remove_fixture_image(docker, tag)
